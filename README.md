# RLT-OpenPI

An implementation of **RL Token: Bootstrapping Online RL with Vision-Language-Action Models** (Xu et al., Physical Intelligence) built on top of [OpenPI](https://github.com/Physical-Intelligence/openpi)'s public checkpoints.

Paper: https://pi.website/research/rlt

![RLT method overview — data, VLA with RL token, online RL, and final RL policy tasks.](docs/rlt_overview.png)

> **Supported environments.** Two backends ship out of the box: a **real Franka Panda + DROID + Oculus VR** rig (`exp/robot_*.sh`) and a **ManiSkill3 simulation** backend (`exp/sim_*.sh`). RLT itself is environment-agnostic — env, intervention manager, data transforms, and VLA checkpoint are all pluggable via `--env-factory`, `--intervention-factory`, `--data-transforms-fn`, and `--vla-config-name`.

---

## Repository Layout

```
src/rlt_openpi/
  models/          RLTokenEncoder/Decoder, Actor (residual), TwinQCritic
  training/        Stage 1 + Stage 2 trainers, configs, replay buffer, TD3 utils
  vla/             OpenPI VLA wrapper, embedding extractor hooks
  rollout/         RolloutWorker, base env/intervention/reward interfaces, factory
  envs/franka/     Example Franka+DROID env factory, VR intervention manager
  envs/maniskill/  ManiSkill3 simulation env factory (DROID-schema obs)
  policies/franka/ Example three-camera DROID data transforms
  utils/           Checkpoint I/O, wandb logger, rich terminal UI
scripts/
  train_rl_token.py    Stage 1 entry point
  train_online_rl.py   Stage 2 entry point
  evaluate.py          Unified Stage 1 / Stage 2 evaluation
  tools/               Data-prep utilities (LeRobot conversion, ManiSkill demo collection, ...)
exp/
  robot_stage1.sh, robot_stage2.sh    Real-robot (Franka+DROID) runs
  sim_stage1.sh, sim_stage2.sh        ManiSkill simulation runs
  eval_vla.sh, eval_full.sh, eval_sim.sh   Evaluation entry points
tests/               Unit tests for models, buffers, and training loop
```

---

## Installation

```bash
git clone https://github.com/yknxh/rlt-openpi.git
cd rlt-openpi
bash setup_env.sh        # creates a conda env named 'rlt'
conda activate rlt
```

The script creates a conda env with Python 3.11, installs OpenPI + rlt-openpi (including [ManiSkill3](https://github.com/haosulab/ManiSkill) for the simulation pipeline) via `uv`, and patches `transformers` with OpenPI's `transformers_replace` files.

You can pass a custom env name: `bash setup_env.sh myenvname`.

### Installation on Robot Machine

On a robot host running Stage 2 with [DROID](https://github.com/droid-dataset/droid), add `--robot` and set `DROID_DIR` to your local DROID clone to also install the DROID teleop stack, Oculus reader, ZED camera bindings, and opencv/protobuf fixups:

```bash
DROID_DIR=/path/to/droid bash setup_env.sh --robot
conda activate rlt
```

Requires the ZED SDK at `/usr/local/zed` for pyzed bindings (skipped gracefully if not found).

> **`.env`**: each `exp/*.sh` auto-sources a project-root `.env` (gitignored). Run `cp .env.example .env` and fill in `WANDB_API_KEY`, `HF_LEROBOT_HOME`, etc. before launching any experiment.

---

## Checkpoints

Download an OpenPI checkpoint from the [OpenPI model zoo](https://github.com/Physical-Intelligence/openpi#checkpoints) (hosted on GCS at `gs://openpi-assets/checkpoints/`). The downloaded JAX/Orbax checkpoint needs to be converted to PyTorch:

```bash
python scripts/tools/convert_jax_to_pytorch.py \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --config-name pi05_droid_finetune \
    --output-path checkpoints/pi05_droid_pytorch
```

This produces a `model.safetensors` file. Point `--train.vla-checkpoint-dir` / `--vla-checkpoint-dir` at it. Any OpenPI checkpoint compatible with your chosen `--vla-config-name` will work.

---

## Data

Stage 1 reads [LeRobot](https://github.com/huggingface/lerobot) datasets. Dataset location is resolved as `$HF_LEROBOT_HOME/<repo_id>/` (default `~/.cache/huggingface/lerobot/`) — set `HF_LEROBOT_HOME` in `.env` and pass the subdirectory as `--repo-id`.

All datasets share the same DROID-style schema: three cameras (`exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`), 7-dim `joint_position`, 1-dim `gripper_position`, and 8-dim `actions`. Two ways to produce one:

**Real-robot demos (Franka/DROID).**
Convert raw DROID HDF5 files → LeRobot, then precompute normalization statistics:
```bash
python scripts/tools/convert_to_lerobot.py --data-dir /path/to/demo_hdf5s --repo-name local/my_task
python scripts/tools/compute_norm_stats.py --repo-id local/my_task
```

**ManiSkill sim demos.**
The built-in motion-planning oracle generates 200 successful episodes for `PegInsertionSide-v1` (default) or `PlugCharger-v1`:
```bash
bash exp/sim_stage1.sh --collect-demos   # runs collect_maniskill_demos.py then trains
```

---

## Stage 1: Train the RL Token

Trains a small encoder/decoder to compress the VLA's internal per-token embeddings `z_{1:M}` into a single **RL token** `z_rl`, via masked MSE reconstruction on a LeRobot demonstration dataset. With `--train.vla-finetune-alpha 0` the VLA is frozen; with `α > 0` the VLA is co-finetuned using a weighted flow-matching loss (matching the paper's `L_ro + α · L_vla` objective).

Example command (see `exp/robot_stage1.sh`, or `exp/sim_stage1.sh` for ManiSkill — pass `--collect-demos` to collect demonstrations first):

```bash
CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/model.safetensors"

python scripts/train_rl_token.py \
    --train.vla-config-name pi05_droid_finetune \
    --train.vla-checkpoint-dir "$CHECKPOINT_DIR" \
    --train.vla-finetune-alpha 1.0 \
    --train.batch-size 32 \
    --train.num-train-steps 5000 \
    --repo-id local/stack_the_blocks_100 \
    --data-transforms-fn rlt_openpi.policies.franka.config.three_camera_droid
```

Swap `--repo-id`, `--data-transforms-fn`, and the VLA config/checkpoint for your own dataset and robot. All flags are documented in `src/rlt_openpi/training/config.py::RLTokenTrainConfig`. Outputs land under `checkpoints/rl_token/<run_name>/rl_token_step<N>.pt`.

---

## Stage 2: Online RL

With VLA + encoder frozen, a lightweight **Actor** and **Twin-Q Critic** are trained online. The actor outputs a residual over the VLA's reference action chunk (zero-initialized last layer, so it starts as a copy of the VLA). After a warmup phase of pure VLA rollouts, training alternates between rollout collection and off-policy TD3 updates with a BC regularizer toward the VLA reference. See the paper for the full algorithm.

### Example command (see `exp/robot_stage2.sh`, or `exp/sim_stage2.sh` for ManiSkill)

```bash
python scripts/train_online_rl.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --intervention-factory rlt_openpi.envs.franka.intervention.make_vr_intervention \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
    --rl-token-checkpoint checkpoints/rl_token/rl_token_step3000.pt \
    --task-prompt "stack the three blocks on the tray" \
    --warmup-steps 250 \
    --chunk-length 5 \
    --max-episode-chunks 150 \
    --save-dir checkpoints/online_rl
```

All flags and hyperparameters are in `src/rlt_openpi/training/config.py::OnlineRLTrainConfig`. Pass a custom `--env-factory` / `--intervention-factory` to run against your own robot or simulator.

---

## Human-in-the-Loop Controls (Stage 2)

**Keyboard rewards** — non-blocking listener in `src/rlt_openpi/rollout/reward.py` reads single keypresses (no Enter needed):

| Key | Meaning |
| --- | --- |
| `s` or `Space` | Success — reward `+1.0`, episode ends. |
| `f` | Failure — reward `0.0`, episode ends. |
| `p` | Progress — reward `+0.5`, episode continues. |

Success/failure are latched; progress is consumed on read. Headless runs (no TTY) degrade gracefully to line-buffered input.

**VR intervention** (example, Franka-specific) — `src/rlt_openpi/envs/franka/intervention.py`. When the operator engages the VR controller, `VRInterventionManager` takes over the current action chunk; the executed human action is written to the replay buffer and downstream BC regularization pulls the actor toward it. On a different rig, provide your own `InterventionManager` subclass.

**Terminal UI** — `src/rlt_openpi/utils/display.py` renders warmup progress, per-episode stats, and operator instructions using [`rich`](https://github.com/Textualize/rich).

---

## Evaluation

`scripts/evaluate.py` auto-detects whether a checkpoint is a Stage 1 (VLA-only) or Stage 2 (VLA + RL token + actor) artifact and runs the appropriate rollout loop on whatever env factory you pass in. Example command (Franka rig):

```bash
# exp/eval_full.sh
python scripts/evaluate.py \
    --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
    --vla-config-name pi05_droid_finetune \
    --vla-checkpoint-dir checkpoints/pi05_droid_pytorch/model.safetensors \
    --rl-token-checkpoint checkpoints/rl_token/rl_token_step5000.pt \
    --checkpoint checkpoints/online_rl/run_latest/online_rl_ep100.pt \
    --task-prompt "stack the three blocks on the tray" \
    --num-episodes 50
```

Results (per-episode success, reward, length) are written to JSON under the run's save dir.

---

## Status & Limitations

This is an **implementation** — unofficial and not affiliated with Physical Intelligence. It is under active development and may still contain bugs.

Both training stages, human-in-the-loop controls, and an evaluation script are wired up end-to-end for the Franka/DROID/VR rig and the ManiSkill sim backend. Not yet validated end-to-end on the four paper tasks (screw installation, zip-tie fastening, Ethernet insertion, charger insertion); other robots and VLA configs are supported in principle but untested here.

---

## Citation

```
@article{xu2025rltoken,
  title   = {RL Token: Bootstrapping Online RL with Vision-Language-Action Models},
  author  = {Xu, Charles and Springenberg, Jost Tobias and Equi, Michael and Amin, Ali
             and Esmail, Adnan and Levine, Sergey and Ke, Liyiming},
  year    = {2025},
  journal = {Physical Intelligence},
  url     = {https://pi.website/research/rlt}
}
```

The VLA backbone and training infrastructure come from [OpenPI](https://github.com/Physical-Intelligence/openpi). All credit for the RLT method belongs to the paper's authors; any errors in this reimplementation are mine.
