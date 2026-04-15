"""Collect ManiSkill demonstrations with motion-planning oracle policies.

Runs the ManiSkill built-in Panda motion-planning solver for a given
task and records each successful episode into a LeRobot v2 dataset
that matches the schema consumed by ``three_camera_droid`` data
transforms.

Usage::

    uv run python scripts/tools/collect_maniskill_demos.py \\
        --env-id PegInsertionSide-v1 \\
        --num-episodes 200 \\
        --repo-name local/maniskill_peg_insertion
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from importlib import import_module
from pathlib import Path

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

# env_id -> (motion-planning solution module, default prompt)
SOLUTION_MAP = {
    "PickCube-v1": (
        "mani_skill.examples.motionplanning.panda.solutions.pick_cube",
        "pick up the red cube",
    ),
    "StackCube-v1": (
        "mani_skill.examples.motionplanning.panda.solutions.stack_cube",
        "stack the red cube on top of the green cube",
    ),
    "PegInsertionSide-v1": (
        "mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side",
        "insert the peg into the hole",
    ),
    "PlugCharger-v1": (
        "mani_skill.examples.motionplanning.panda.solutions.plug_charger",
        "plug the charger into the wall receptacle",
    ),
}


@dataclasses.dataclass
class CollectConfig:
    env_id: str = "PegInsertionSide-v1"
    num_episodes: int = 200
    repo_name: str = "local/maniskill_peg_insertion"
    task_prompt: str = ""
    image_width: int = 320  # DROID convention
    image_height: int = 180
    # Env is run in pd_joint_pos (required by ManiSkill motion-planning solvers),
    # but stored actions have arm dims converted to deltas below so the dataset
    # matches the LeRobot DROID pipeline's delta assumption.
    control_mode: str = "pd_joint_pos"
    max_episode_steps: int = 200
    fps: int = 15  # match DROID recording fps
    control_freq: int = 15  # match DROID control rate (15 Hz)
    seed: int = 0
    overwrite: bool = False


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _get_camera_image(obs: dict, cam_name: str) -> np.ndarray | None:
    sensors = obs.get("sensor_data", {})
    if cam_name in sensors and sensors[cam_name].get("rgb") is not None:
        return _to_numpy(sensors[cam_name]["rgb"]).astype(np.uint8)
    return None


def main(cfg: CollectConfig) -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    if cfg.env_id not in SOLUTION_MAP:
        raise ValueError(
            f"No motion-planning solution registered for {cfg.env_id}. "
            f"Available: {list(SOLUTION_MAP)}"
        )
    solution_module_path, default_prompt = SOLUTION_MAP[cfg.env_id]
    solve_module = import_module(solution_module_path)
    solve_fn = getattr(solve_module, "solve")
    task_prompt = cfg.task_prompt or default_prompt

    output_path = HF_LEROBOT_HOME / cfg.repo_name
    if output_path.exists():
        if cfg.overwrite:
            log.info("Removing existing dataset at %s", output_path)
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Dataset already exists at {output_path}. Use --overwrite to replace."
            )

    img_shape = (cfg.image_height, cfg.image_width, 3)
    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_name,
        robot_type="panda",
        fps=cfg.fps,
        features={
            "exterior_image_1_left": {"dtype": "image", "shape": img_shape, "names": ["height", "width", "channel"]},
            "wrist_image_left": {"dtype": "image", "shape": img_shape, "names": ["height", "width", "channel"]},
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        image_writer_threads=4,
        image_writer_processes=4,
    )

    sim_freq = max(120, cfg.control_freq * 8)
    sim_freq -= sim_freq % cfg.control_freq

    env = gym.make(
        cfg.env_id,
        obs_mode="rgb",
        control_mode=cfg.control_mode,
        sim_config=dict(sim_freq=sim_freq, control_freq=cfg.control_freq),
        render_mode="rgb_array",
        sensor_configs=dict(width=cfg.image_width, height=cfg.image_height),
        max_episode_steps=cfg.max_episode_steps,
    )

    # ManiSkill motion-planning solvers drive the env internally and return
    # only a status dict — they don't hand back (obs, action) trajectories.
    # Wrap reset/step so we can record the trajectory while the solver runs.
    recorded_obs: list = []
    recorded_actions: list = []
    orig_reset = env.reset
    orig_step = env.step

    def reset_hook(**kw):
        out = orig_reset(**kw)
        recorded_obs.clear()
        recorded_actions.clear()
        obs0 = out[0] if isinstance(out, tuple) else out
        recorded_obs.append(obs0)
        return out

    def step_hook(action):
        recorded_actions.append(np.asarray(_to_numpy(action)))
        out = orig_step(action)
        recorded_obs.append(out[0])
        return out

    env.reset = reset_hook  # type: ignore[assignment]
    env.step = step_hook  # type: ignore[assignment]

    saved = 0
    attempts = 0
    pbar = tqdm(total=cfg.num_episodes, desc="Collecting demos")
    while saved < cfg.num_episodes:
        seed = cfg.seed + attempts
        attempts += 1
        recorded_obs.clear()
        recorded_actions.clear()
        try:
            res = solve_fn(env, seed=seed, debug=False, vis=False)
        except Exception as e:  # noqa: BLE001
            log.warning("Solver raised on seed %d: %s", seed, e)
            continue

        success = False
        if isinstance(res, dict):
            success = bool(_to_numpy(res.get("success", False)))
        elif isinstance(res, tuple):
            # Some solvers return (info, success) or similar.
            last = res[-1]
            success = bool(_to_numpy(last)) if not isinstance(last, dict) else bool(
                _to_numpy(last.get("success", False))
            )
        if not success:
            continue

        n_steps = min(len(recorded_actions), len(recorded_obs) - 1)
        if n_steps == 0:
            log.warning("Solver recorded zero steps on seed %d; skipping", seed)
            continue

        for t in range(n_steps):
            obs_t = recorded_obs[t]
            action_t = _to_numpy(recorded_actions[t]).astype(np.float32)
            if action_t.shape[0] != 8:
                # Control-mode mismatch; abort this episode.
                log.warning(
                    "Expected 8-dim action, got %d. Check control_mode.", action_t.shape[0]
                )
                break
            qpos = _to_numpy(obs_t["agent"]["qpos"]).astype(np.float32)
            joint_pos = qpos[:7]
            # Gripper state: remap ManiSkill finger displacement (~[0, 0.04] m)
            # to DROID-style scalar in [0, 1] where 0=open, 1=closed.
            gripper_pos = np.asarray(
                [float(np.clip(1.0 - qpos[7] / 0.04, 0.0, 1.0))], dtype=np.float32
            )

            # Arm dims become deltas (target - current) to match the LeRobot
            # DROID pipeline's velocity/delta assumption. Gripper action gets
            # remapped from ManiSkill's [-1, 1] (open..close) to DROID [0, 1].
            action_t = action_t.copy()
            action_t[:7] = action_t[:7] - joint_pos
            action_t[7] = float(np.clip(0.5 * (action_t[7] + 1.0), 0.0, 1.0))

            base_img = _get_camera_image(obs_t, "base_camera")
            hand_img = _get_camera_image(obs_t, "hand_camera")
            if base_img is None or hand_img is None:
                raise RuntimeError(
                    "Expected base_camera and hand_camera in observation; "
                    "check env supports them (e.g. uses PandaWristCam agent)."
                )

            dataset.add_frame(
                {
                    "exterior_image_1_left": base_img,
                    "wrist_image_left": hand_img,
                    "joint_position": joint_pos,
                    "gripper_position": gripper_pos,
                    "actions": action_t,
                    "task": task_prompt,
                }
            )

        for key in dataset.features:
            if dataset.features[key].get("shape") == (1,):
                buf = dataset.episode_buffer.get(key)
                if isinstance(buf, list):
                    dataset.episode_buffer[key] = [
                        v.item() if isinstance(v, np.ndarray) else v for v in buf
                    ]
        dataset.save_episode()
        saved += 1
        pbar.update(1)

    pbar.close()
    env.close()
    log.info("Saved %d episodes (after %d attempts) to %s", saved, attempts, output_path)


if __name__ == "__main__":
    main(tyro.cli(CollectConfig))
