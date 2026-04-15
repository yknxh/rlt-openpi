#!/usr/bin/env bash
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a

CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors"

python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.maniskill.env_factory.make_maniskill_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir "$CHECKPOINT_DIR" \
    --rl-token-checkpoint checkpoints/rl_token/rl_token_step2000.pt \
    --checkpoint checkpoints/online_rl_sim/run_latest/online_rl_ep100.pt \
    --task-prompt "insert the peg into the hole" \
    --action-dim 8 \
    --chunk-length 5 \
    --env-kwargs-json '{"env_id": "PegInsertionSide-v1"}' \
    --num-episodes 50
