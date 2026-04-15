"""Configuration dataclasses for RLT-OpenPI training stages."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RLTokenTrainConfig:
    """Stage 1: RL token encoder-decoder training hyperparameters."""

    # Architecture
    embedding_dim: int = 2048
    encoder_layers: int = 2
    encoder_heads: int = 8
    decoder_layers: int = 2
    decoder_heads: int = 8

    # Training
    num_train_steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500  # Linear LR warmup steps (matches OpenPI default)
    max_grad_norm: float = 1.0  # Global gradient norm clipping (matches OpenPI default)
    vla_finetune_alpha: float = 0.0  # VLA fine-tuning weight (0 = frozen VLA)
    vla_learning_rate: float = 1e-5  # VLA fine-tuning learning rate (used when alpha > 0)
    gradient_checkpointing: bool = True  # Enable gradient checkpointing to reduce VRAM

    # Checkpoints
    vla_checkpoint_dir: str = ""
    vla_config_name: str = "pi05_droid_finetune"
    resume_checkpoint: str = ""  # Path to Stage 1 checkpoint to resume training from
    save_dir: str = "checkpoints/rl_token"
    run_name: str = ""  # Subdirectory name for this run (auto-generated if empty)
    save_every: int = 1000
    log_every: int = 1  # wandb logging interval (steps)
    print_every: int = 100  # stdout logging interval (steps)

    # wandb
    wandb_project: str = "rlt-openpi"
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")


@dataclass
class OnlineRLTrainConfig:
    """Stage 2: Online RL training hyperparameters (Algorithm 1)."""

    # Architecture (shared by RLTokenTrainConfig — must match the Stage 1 model)
    embedding_dim: int = 2048

    # Action space
    action_dim: int = 8
    chunk_length: int = 10  # C
    vla_action_horizon: int = 16  # H: number of action steps the VLA outputs

    # Actor-critic architecture
    mlp_hidden_dim: int = 256
    mlp_num_hidden_layers: int = 2
    actor_noise_sigma: float = 0.1  # actor exploration noise std
    ref_action_dropout: float = 0.5

    # RL hyperparameters
    gamma: float = 0.99
    tau: float = 0.005  # Polyak averaging coefficient
    utd_ratio: int = 5  # G: update-to-data ratio
    bc_regularizer_beta: float = 0.5  # BC regularizer coefficient
    critic_updates_per_actor: int = 2
    target_noise_sigma: float = 0.2  # TD3 target policy smoothing noise std
    target_noise_clip: float = 0.5  # clamp range for target noise

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Replay buffer
    buffer_capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1000

    # Environment
    env_factory: str = ""  # Python import path, e.g. "rlt_openpi.envs.franka.env_factory.make_franka_env"
    intervention_factory: str = ""  # Python import path, e.g. "rlt_openpi.envs.franka.intervention.make_vr_intervention"
    task_prompt: str = ""  # Task instruction for VLA (passed to env factory)
    max_episode_chunks: int = 150  # Max chunks per episode before forced termination
    env_kwargs_json: str = ""  # Extra env-factory kwargs, JSON-encoded (merged into make_env call)

    # Training loop
    max_env_steps: int = 100_000

    # Checkpoints
    rl_token_checkpoint: str = ""
    vla_checkpoint_dir: str = ""
    vla_config_name: str = "pi05_droid_finetune"
    resume_checkpoint: str = ""  # Path to Stage 2 checkpoint to resume training from
    warmup_buffer: str = ""  # Path to a standalone warmup buffer .pt file (skips warmup if provided)
    save_dir: str = "checkpoints/online_rl"
    run_name: str = ""  # Subdirectory name for this run (auto-generated if empty)
    save_every: int = 50
    log_every: int = 1  # wandb logging interval (steps)
    print_every: int = 100  # stdout logging interval (steps)

    # wandb
    wandb_project: str = "rlt-openpi"
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    @property
    def state_dim(self) -> int:
        """RL state dimension: z_rl (embedding_dim) + s^p (action_dim)."""
        return self.embedding_dim + self.action_dim

    @property
    def action_chunk_dim(self) -> int:
        """Flattened action chunk dimension: C * d."""
        return self.chunk_length * self.action_dim
