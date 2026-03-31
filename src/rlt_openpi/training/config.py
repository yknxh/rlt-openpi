"""Configuration dataclasses for RLT-OpenPI training stages."""

from dataclasses import dataclass


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
    vla_finetune_alpha: float = 0.0  # VLA fine-tuning weight (0 = frozen VLA)

    # Checkpoints
    vla_checkpoint_dir: str = ""
    vla_config_name: str = "pi0_aloha_sim"
    save_dir: str = "checkpoints/rl_token"
    save_every: int = 1000
    log_every: int = 100

    # wandb
    wandb_project: str = "rlt-openpi"
    wandb_enabled: bool = True


@dataclass
class OnlineRLTrainConfig:
    """Stage 2: Online RL training hyperparameters (Algorithm 1)."""

    # Architecture (shared by RLTokenTrainConfig — must match the Stage 1 model)
    embedding_dim: int = 2048

    # Action space
    action_dim: int = 14
    chunk_length: int = 10  # C
    vla_action_horizon: int = 50  # H: number of action steps the VLA outputs

    # Actor-critic architecture
    mlp_hidden_dim: int = 256
    mlp_num_hidden_layers: int = 2
    actor_noise_sigma: float = 0.1  # actor exploration noise std
    ref_action_dropout: float = 0.5

    # RL hyperparameters
    gamma: float = 0.99
    tau: float = 0.005  # Polyak averaging coefficient
    utd_ratio: int = 5  # G: update-to-data ratio
    bc_regularizer_beta: float = 1.0  # BC regularizer coefficient
    critic_updates_per_actor: int = 2

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Replay buffer
    buffer_capacity: int = 100_000
    warmup_steps: int = 1000

    # Training loop
    max_env_steps: int = 500_000

    # Checkpoints
    rl_token_checkpoint: str = ""
    vla_checkpoint_dir: str = ""
    vla_config_name: str = "pi0_aloha_sim"
    save_dir: str = "checkpoints/online_rl"
    save_every: int = 5000
    log_every: int = 100

    # wandb
    wandb_project: str = "rlt-openpi"
    wandb_enabled: bool = True

    @property
    def state_dim(self) -> int:
        """RL state dimension: z_rl (embedding_dim) + s^p (action_dim)."""
        return self.embedding_dim + self.action_dim

    @property
    def action_chunk_dim(self) -> int:
        """Flattened action chunk dimension: C * d."""
        return self.chunk_length * self.action_dim
