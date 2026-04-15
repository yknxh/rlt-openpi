#!/bin/bash
# Evaluate the full trained model (Stage 1 RL token + Stage 2 actor).
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a

python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
    --rl-token-checkpoint checkpoints/rl_token/rl_token_step5000.pt \
    --checkpoint checkpoints/online_rl/run_latest/online_rl_ep100.pt \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 50
