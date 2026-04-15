#!/bin/bash
# Evaluate the Stage 1 fine-tuned VLA (no RL token head, no actor).
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a

CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors"

# --- Robot (real Franka) ---
python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir "$CHECKPOINT_DIR" \
    --stage1-checkpoint checkpoints/rl_token/rl_token_step3000.pt \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 10 \
    --save-dir results/vla_only

# --- Sim (ManiSkill) ---
# python scripts/evaluate.py \
#     --env-factory rlt_openpi.envs.maniskill.env_factory.make_maniskill_env \
#     --vla-config-name pi05_droid_finetune \
#     --vla-checkpoint-dir "$CHECKPOINT_DIR" \
#     --stage1-checkpoint checkpoints/rl_token/sim_stage1/rl_token_step5000.pt \
#     --task-prompt "insert the peg into the hole" \
#     --env-kwargs-json '{"env_id": "PegInsertionSide-v1", "record_video_dir": "results/sim_stage1_eval/videos"}' \
#     --num-episodes 50 \
#     --save-dir results/sim_stage1_eval   
