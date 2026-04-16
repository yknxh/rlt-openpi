"""Environment factory for ManiSkill3 Franka Panda tasks.

Wraps a ManiSkill gymnasium env so its observations are remapped to
DROID-schema keys (3 cameras + joint/gripper state). This lets the
existing ``three_camera_droid`` data transforms and ``pi05_droid``
VLA checkpoint run in simulation without changes.

Install ManiSkill with ``pip install -e ".[sim]"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rlt_openpi.rollout.sim_env import SimEnv


class ManiSkillSimEnv(SimEnv):
    """SimEnv subclass that emits DROID-schema observation dicts."""

    def __init__(
        self,
        env,
        action_dim: int,
        chunk_length: int,
        task_prompt: str,
        max_episode_chunks: int,
    ) -> None:
        super().__init__(env=env, action_dim=action_dim, chunk_length=chunk_length)
        self._task_prompt = task_prompt
        self._max_episode_chunks = max_episode_chunks
        self._chunks_executed = 0
        self._current_qpos: NDArray | None = None

    @property
    def max_episode_chunks(self) -> int:
        return self._max_episode_chunks

    def _to_numpy(self, x: Any) -> NDArray:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        arr = np.asarray(x)
        # ManiSkill batches by default: drop leading batch dim of size 1.
        if arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def _get_camera_image(self, obs: dict, cam_name: str) -> NDArray:
        sensors = obs.get("sensor_data", {})
        if cam_name not in sensors:
            return None
        rgb = sensors[cam_name].get("rgb")
        if rgb is None:
            return None
        img = self._to_numpy(rgb)
        return img.astype(np.uint8)

    def _make_obs_dict(self, obs: Any) -> dict[str, Any]:
        qpos = self._to_numpy(obs["agent"]["qpos"])
        # Panda qpos is 9-dim (7 arm + 2 gripper fingers).
        joint_position = np.asarray(qpos[:7], dtype=np.float32)
        # Remap ManiSkill finger displacement (~[0, 0.04] m) to DROID-style
        # scalar in [0, 1] where 0=open, 1=closed. Must match the convention
        # used when collecting demos.
        gripper_position = np.asarray(
            [float(np.clip(1.0 - qpos[7] / 0.04, 0.0, 1.0))], dtype=np.float32
        )

        base_img = self._get_camera_image(obs, "base_camera")
        hand_img = self._get_camera_image(obs, "hand_camera")
        right_img = self._get_camera_image(obs, "right_camera")

        out: dict[str, Any] = {
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
            "observation/exterior_image_1_left": base_img,
            "observation/wrist_image_left": hand_img,
            "prompt": self._task_prompt,
        }
        # Only include the 3rd-view slot if a real camera exists. Otherwise
        # ThreeCameraDroidInputs zero-fills it with mask=False, matching the
        # 2-real + 1-masked setup pi05_droid was pretrained on.
        if right_img is not None:
            out["observation/exterior_image_2_left"] = right_img
        return out

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        self._chunks_executed = 0
        obs, _info = self.env.reset(**kwargs)
        self._current_qpos = self._to_numpy(obs["agent"]["qpos"])
        return self._make_obs_dict(obs)

    def step(self, action_chunk: NDArray):
        C = self._chunk_length
        rewards = np.zeros(C, dtype=np.float32)
        done = False
        info: dict[str, Any] = {}
        obs = None
        k = 0

        for k in range(C):
            step_action = np.asarray(action_chunk[k], dtype=np.float32).copy()
            # VLA emits arm deltas and gripper in DROID-style [0, 1].
            # Convert arm deltas to absolute joint position targets (matching
            # pd_joint_pos used during demo collection) and remap gripper
            # from DROID [0, 1] to ManiSkill [-1, 1].
            step_action[:7] = self._current_qpos[:7] + step_action[:7]
            step_action[7] = float(np.clip(2.0 * step_action[7] - 1.0, -1.0, 1.0))
            obs, reward, terminated, truncated, info = self.env.step(step_action)
            self._current_qpos = self._to_numpy(obs["agent"]["qpos"])
            rewards[k] = float(self._to_numpy(reward))
            term = bool(self._to_numpy(terminated))
            trunc = bool(self._to_numpy(truncated))
            done = term or trunc
            if done:
                break

        self._chunks_executed += 1
        if self._chunks_executed >= self._max_episode_chunks:
            done = True

        info_out: dict[str, Any] = {"steps_executed": k + 1}
        success = info.get("success")
        if success is not None:
            info_out["success"] = bool(self._to_numpy(success))
        fail = info.get("fail")
        if fail is not None:
            info_out["fail"] = bool(self._to_numpy(fail))

        assert obs is not None
        return self._make_obs_dict(obs), rewards, done, info_out


def make_maniskill_env(
    action_dim: int = 8,
    chunk_length: int = 10,
    task_prompt: str = "insert the peg into the hole",
    env_id: str = "PegInsertionSide-v1",
    max_episode_steps: int = 100,
    max_episode_chunks: int = 50,
    image_width: int = 320,
    image_height: int = 180,
    control_mode: str = "pd_joint_pos",
    control_freq: int = 15,  # match DROID control rate
    reward_mode: str = "normalized_dense",
    record_video_dir: str = "",
    **kwargs,
) -> ManiSkillSimEnv:
    """Create a ManiSkill env wrapped as a DROID-schema ``SimEnv``.

    Args:
        action_dim: Expected single-step action dimension (8 for Panda +
            gripper under ``pd_joint_delta_pos``).
        chunk_length: Number of single-step actions per RL chunk.
        task_prompt: Language instruction passed through to the VLA.
        env_id: ManiSkill task id (``PickCube-v1``, ``StackCube-v1``, ...).
        max_episode_steps: Per-environment step cap (gymnasium TimeLimit).
        max_episode_chunks: Per-env-wrapper cap in number of chunks.
        image_width: Camera image width (default 320 — matches DROID convention).
        image_height: Camera image height (default 180 — matches DROID convention).
        control_mode: ManiSkill control mode. ``pd_joint_pos`` (default)
            matches the controller used during demo collection. The wrapper
            converts VLA delta outputs to absolute position targets before
            stepping. Must match the control mode used at demo-collection time.
        reward_mode: ManiSkill reward mode. One of:
            - ``"normalized_dense"`` (default): shaped reward in ~[0, 1] per step.
            - ``"dense"``: shaped reward, un-normalized (raw magnitude).
            - ``"sparse"``: 0 until success, then 1.
            - ``"none"``: always 0 (for pure imitation / no-reward debugging).
        record_video_dir: If non-empty, wraps the env with ManiSkill's
            ``RecordEpisode`` and writes one ``.mp4`` per episode to this dir.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers envs with gymnasium

    # control_freq must divide sim_freq; bump sim_freq off its default (100)
    # to a multiple of control_freq.
    sim_freq = max(120, control_freq * 8)
    sim_freq -= sim_freq % control_freq

    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode=control_mode,
        sim_config=dict(sim_freq=sim_freq, control_freq=control_freq),
        reward_mode=reward_mode,
        render_mode="rgb_array",
        sensor_configs=dict(width=image_width, height=image_height),
        max_episode_steps=max_episode_steps,
    )

    if record_video_dir:
        from mani_skill.utils.wrappers import RecordEpisode

        env = RecordEpisode(
            env,
            output_dir=record_video_dir,
            save_video=True,
            save_trajectory=False,
            video_fps=30,
        )

    return ManiSkillSimEnv(
        env=env,
        action_dim=action_dim,
        chunk_length=chunk_length,
        task_prompt=task_prompt,
        max_episode_chunks=max_episode_chunks,
    )
