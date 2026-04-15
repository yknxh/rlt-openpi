#!/usr/bin/env bash
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a

CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors"

python scripts/train_rl_token.py \
    --train.vla-config-name pi05_droid_finetune \
    --train.vla-checkpoint-dir "$CHECKPOINT_DIR" \
    --train.vla-finetune-alpha 1.0 \
    --train.batch-size 32 \
    --train.num-train-steps 3000 \
    --train.warmup-steps 200 \
    --repo-id local/put_the_pet_demo \
    --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
