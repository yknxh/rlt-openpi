#!/usr/bin/env bash
# Stage 1 RL-token training on ManiSkill demonstrations.
#
# Usage:
#   exp/sim_stage1.sh                  # train on existing dataset
#   exp/sim_stage1.sh --collect-demos  # collect 200 demos first, then train
set -euo pipefail

# Auto-load project-root .env (WANDB_API_KEY, HF_LEROBOT_HOME, ...)
[ -f .env ] && set -a && . ./.env && set +a

CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors"
ENV_ID="PegInsertionSide-v1"
REPO_ID="local/maniskill_peg_insertion"
NUM_DEMOS=200

COLLECT_DEMOS=0
for arg in "$@"; do
    case "$arg" in
        --collect-demos) COLLECT_DEMOS=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

if [[ "$COLLECT_DEMOS" -eq 1 ]]; then
    echo "[sim_stage1] Collecting $NUM_DEMOS demos for $ENV_ID -> $REPO_ID"
    python scripts/tools/collect_maniskill_demos.py \
        --env-id "$ENV_ID" \
        --num-episodes "$NUM_DEMOS" \
        --repo-name "$REPO_ID" \
        --overwrite
fi

python scripts/train_rl_token.py \
    --train.vla-config-name pi05_droid_finetune \
    --train.vla-checkpoint-dir "$CHECKPOINT_DIR" \
    --train.vla-finetune-alpha 1.0 \
    --train.batch-size 32 \
    --train.num-train-steps 4000 \
    --train.warmup-steps 200 \
    --repo-id "$REPO_ID" \
    --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
