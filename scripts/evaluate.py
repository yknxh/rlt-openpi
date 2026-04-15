"""Evaluate a trained model on an environment.

Supports two modes:
  - **stage1**: VLA-only evaluation (fine-tuned VLA from Stage 1, no actor).
  - **stage2**: Full pipeline (VLA + RL token + actor from Stage 2).

The mode is auto-detected: if --checkpoint (Stage 2) is provided, runs
stage2 eval; otherwise runs stage1 VLA-only eval.

Usage:
    # Stage 1: evaluate fine-tuned VLA
    uv run python scripts/evaluate.py \
        --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
        --stage1-checkpoint checkpoints/rl_token/rl_token_step5000.pt \
        --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
        --task-prompt "stack the three blocks on the tray" \
        --num-episodes 50

    # Stage 2: evaluate full trained model
    uv run python scripts/evaluate.py \
        --checkpoint checkpoints/online_rl/run_latest/online_rl_ep100.pt \
        --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
        --rl-token-checkpoint checkpoints/rl_token/rl_token_step5000.pt \
        --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
        --task-prompt "stack the three blocks on the tray" \
        --num-episodes 50
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro
from tqdm import tqdm

from rlt_openpi.models.actor import Actor
from rlt_openpi.rollout.factory import make_env
from rlt_openpi.rollout.intervention import InterventionManager
from rlt_openpi.rollout.rollout_worker import RolloutWorker
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.replay_buffer import ReplayBuffer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def _parse_env_kwargs(raw: str) -> dict:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--env-kwargs-json must decode to a JSON object")
    return parsed


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Stage 2 checkpoint (if provided, runs full pipeline eval)
    checkpoint: str = ""
    # Stage 1 checkpoint with fine-tuned VLA weights (if provided without
    # --checkpoint, runs VLA-only eval)
    stage1_checkpoint: str = ""

    vla_config_name: str = "pi05_droid_finetune"
    vla_checkpoint_dir: str = ""
    rl_token_checkpoint: str = ""
    env_factory: str = ""
    env_kwargs_json: str = ""  # Extra env-factory kwargs, JSON-encoded
    task_prompt: str = ""
    action_dim: int = 8
    chunk_length: int = 10
    num_episodes: int = 50
    save_dir: str = ""
    device: str = "cuda"


def _run(config: EvalConfig) -> None:
    """Evaluate full pipeline: VLA + RL token + actor."""
    ckpt = torch.load(config.checkpoint, map_location=config.device, weights_only=False)
    train_config: OnlineRLTrainConfig = ckpt["config"]

    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device=config.device,
    )

    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device=config.device)
    rl_token_model.eval()

    actor = Actor(
        state_dim=train_config.state_dim,
        action_chunk_dim=train_config.action_chunk_dim,
        hidden_dim=train_config.mlp_hidden_dim,
        num_hidden_layers=train_config.mlp_num_hidden_layers,
        sigma=train_config.actor_noise_sigma,
        ref_dropout=0.0,
    ).to(config.device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    log.info(
        "Actor loaded from %s (episode %d, %d env steps)",
        config.checkpoint,
        ckpt["total_episodes"],
        ckpt["total_env_steps"],
    )

    env = make_env(
        config.env_factory,
        action_dim=train_config.action_dim,
        chunk_length=train_config.chunk_length,
        task_prompt=config.task_prompt,
        **_parse_env_kwargs(config.env_kwargs_json),
    )
    log.info("Environment created: action_dim=%d, chunk_length=%d", env.action_dim, env.chunk_length)

    buf = ReplayBuffer(1, train_config.state_dim, train_config.action_chunk_dim, train_config.chunk_length)
    worker = RolloutWorker(
        env, vla, rl_token_model, actor, buf,
        InterventionManager(), train_config.chunk_length,
        train_config.action_dim, config.device,
    )

    episodes = []
    for ep in range(config.num_episodes):
        stats = worker.collect_episode(store_transitions=False)
        success = stats.extra.get("success", False)
        episodes.append({
            "episode": ep,
            "reward": stats.total_reward,
            "success": success,
            "num_chunks": stats.num_chunks,
            "num_steps": stats.num_steps,
        })
        log.info("Episode %d: reward=%.3f, success=%s", ep, stats.total_reward, success)

    return episodes, {
        "mode": "stage2",
        "checkpoint": config.checkpoint,
        "train_episodes": ckpt["total_episodes"],
        "train_env_steps": ckpt["total_env_steps"],
    }


def _run_vla(config: EvalConfig) -> None:
    """Evaluate VLA-only (with optional Stage 1 fine-tuned weights)."""
    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device=config.device,
    )

    if config.stage1_checkpoint:
        ckpt = torch.load(config.stage1_checkpoint, map_location="cpu", weights_only=False)
        if "vla_model" in ckpt:
            vla.extractor.pi0.load_state_dict(ckpt["vla_model"])
            log.info("Loaded fine-tuned VLA weights from %s", config.stage1_checkpoint)
        else:
            log.warning("No vla_model key in %s; using base VLA weights", config.stage1_checkpoint)
        del ckpt

    env = make_env(
        config.env_factory,
        action_dim=config.action_dim,
        chunk_length=config.chunk_length,
        task_prompt=config.task_prompt,
        **_parse_env_kwargs(config.env_kwargs_json),
    )
    log.info("Environment created: action_dim=%d, chunk_length=%d", env.action_dim, env.chunk_length)

    episodes = []
    pbar = tqdm(range(config.num_episodes), desc="VLA eval", unit="ep")
    num_success = 0
    for ep in pbar:
        obs = env.reset()
        episode_reward = 0.0
        episode_chunks = 0

        chunk_pbar = tqdm(desc=f"ep {ep+1}", unit="chunk", leave=False)
        while True:
            with torch.no_grad():
                vla_input = vla.preprocess_obs(obs)
                action_chunk = vla.get_rl_chunk_reference(vla_input, config.chunk_length)
                action_chunk = action_chunk.squeeze(0).cpu().numpy()

            next_obs, chunk_rewards, done, info = env.step(action_chunk)
            episode_reward += float(chunk_rewards.sum())
            episode_chunks += 1
            chunk_pbar.update(1)
            chunk_pbar.set_postfix(reward=f"{episode_reward:.2f}")

            if done:
                success = info.get("success", False)
                num_success += int(bool(success))
                episodes.append({
                    "episode": ep,
                    "reward": episode_reward,
                    "success": success,
                    "num_chunks": episode_chunks,
                    "num_steps": info.get("steps_executed", episode_chunks * config.chunk_length),
                })
                chunk_pbar.close()
                pbar.set_postfix(
                    success_rate=f"{num_success}/{ep+1}",
                    last_reward=f"{episode_reward:.2f}",
                )
                log.info(
                    "Episode %d/%d: chunks=%d, reward=%.3f, success=%s",
                    ep + 1, config.num_episodes, episode_chunks, episode_reward, success,
                )
                break

            obs = next_obs

    return episodes, {
        "mode": "stage1",
        "stage1_checkpoint": config.stage1_checkpoint,
    }


def main(config: EvalConfig) -> None:
    """Run evaluation (auto-detects stage1 vs stage2)."""
    log.info("Eval config: %s", config)

    if not config.env_factory:
        log.error("--env-factory is required.")
        raise SystemExit(1)

    if config.checkpoint:
        log.info("Full eval: VLA + RL token + actor")
        episodes, meta = _run(config)
    else:
        log.info("VLA-only eval")
        episodes, meta = _run_vla(config)

    num_success = sum(e["success"] for e in episodes)
    success_rate = num_success / len(episodes) if episodes else 0.0
    mean_reward = sum(e["reward"] for e in episodes) / len(episodes) if episodes else 0.0

    log.info("Success rate: %.1f%% (%d/%d)", 100 * success_rate, num_success, len(episodes))
    log.info("Mean reward: %.3f", mean_reward)

    results = {
        **meta,
        "vla_checkpoint_dir": config.vla_checkpoint_dir,
        "vla_config_name": config.vla_config_name,
        "eval_timestamp": datetime.now().isoformat(),
        "num_episodes": len(episodes),
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "episodes": episodes,
    }

    if config.save_dir:
        save_dir = Path(config.save_dir)
    elif config.checkpoint:
        save_dir = Path(config.checkpoint).parent
    else:
        save_dir = Path("results") / meta["mode"]
    save_dir.mkdir(parents=True, exist_ok=True)

    results_path = save_dir / f"eval_{meta['mode']}_{len(episodes)}ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
