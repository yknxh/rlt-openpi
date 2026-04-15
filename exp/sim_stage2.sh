#!/usr/bin/env bash
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a

CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors"

python scripts/train_online_rl.py \
    --env-factory rlt_openpi.envs.maniskill.env_factory.make_maniskill_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir "$CHECKPOINT_DIR" \
    --rl-token-checkpoint checkpoints/rl_token/sim_stage1/rl_token_step5000.pt \
    --task-prompt "insert the peg into the hole" \
    --action-dim 8 \
    --chunk-length 5 \
    --warmup-steps 250 \
    --max-episode-chunks 50 \
    --env-kwargs-json '{"env_id": "PegInsertionSide-v1"}' \
    --save-dir checkpoints/online_rl_sim
