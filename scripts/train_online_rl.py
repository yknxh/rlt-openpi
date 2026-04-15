"""Stage 2: Online RL training with frozen VLA + RL token (Algorithm 1).

Usage:
    uv run python scripts/train_online_rl.py --help
    uv run python scripts/train_online_rl.py --vla-config-name pi0_aloha_sim \
        --vla-checkpoint-dir /path/to/vla.safetensors \
        --rl-token-checkpoint /path/to/rl_token.pt
"""

from __future__ import annotations

import json
import logging

import torch
import tyro

from rlt_openpi.rollout.factory import make_env, make_intervention
from rlt_openpi.rollout.intervention import InterventionManager
from rlt_openpi.training.config import OnlineRLTrainConfig
from rlt_openpi.training.online_rl_trainer import OnlineRLTrainer
from rlt_openpi.utils.checkpoint import load_rl_token_model
from rlt_openpi.utils.logging import Logger
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def main(config: OnlineRLTrainConfig) -> None:
    """Run online RL training (Stage 2, Algorithm 1)."""
    log.info("Stage 2 config: %s", config)

    # Set up logger
    rl_logger = Logger.from_train_config(config)

    # Load frozen VLA
    log.info("Loading VLA: config=%s, checkpoint=%s", config.vla_config_name, config.vla_checkpoint_dir)
    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device="cuda",
    )

    # Load frozen RL token model from Stage 1
    log.info("Loading RL token model from %s", config.rl_token_checkpoint)
    rl_token_model = load_rl_token_model(config.rl_token_checkpoint, device="cuda")

    # Restore fine-tuned VLA weights from Stage 1 checkpoint (if available).
    # Load to CPU first to avoid OOM — the VLA + RL token already occupy most VRAM.
    stage1_ckpt = torch.load(config.rl_token_checkpoint, map_location="cpu", weights_only=False)
    if "vla_model" in stage1_ckpt:
        vla.extractor.pi0.load_state_dict(stage1_ckpt["vla_model"])
        log.info("Restored fine-tuned VLA weights from Stage 1 checkpoint")
    else:
        log.warning("No fine-tuned VLA weights found in Stage 1 checkpoint; using base VLA")
    del stage1_ckpt
    torch.cuda.empty_cache()

    # Create trainer
    trainer = OnlineRLTrainer(
        config=config,
        vla=vla,
        rl_token_model=rl_token_model,
        device="cuda",
    )

    # Resume from checkpoint if provided
    if config.resume_checkpoint:
        log.info("Resuming from checkpoint: %s", config.resume_checkpoint)
        trainer.load(config.resume_checkpoint)

    # Create environment via pluggable factory.
    # Pass --env-factory to specify a Python import path, e.g.:
    #   --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env
    #   --env-factory rlt_openpi.rollout.sim_env.make_sim_env
    if not config.env_factory:
        log.error("--env-factory is required. Provide a Python import path to an env factory function.")
        raise SystemExit(1)

    extra_env_kwargs: dict = {}
    if config.env_kwargs_json:
        extra_env_kwargs = json.loads(config.env_kwargs_json)
        if not isinstance(extra_env_kwargs, dict):
            raise ValueError("--env-kwargs-json must decode to a JSON object")

    env = make_env(
        config.env_factory,
        action_dim=config.action_dim,
        chunk_length=config.chunk_length,
        task_prompt=config.task_prompt,
        max_episode_chunks=config.max_episode_chunks,
        **extra_env_kwargs,
    )
    log.info("Environment created: action_dim=%d, chunk_length=%d", env.action_dim, env.chunk_length)

    # Create intervention manager (VR teleoperation, etc.) if specified.
    intervention_mgr: InterventionManager | None = None
    if config.intervention_factory:
        intervention_mgr = make_intervention(config.intervention_factory, env=env)
        log.info("Intervention manager created via %s", config.intervention_factory)

    trainer.train(env=env, intervention_mgr=intervention_mgr, log_fn=rl_logger.log)

    rl_logger.finish()


if __name__ == "__main__":
    main(tyro.cli(OnlineRLTrainConfig))
