"""Hyperparameters for training and testing."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class HParams:
    """Common hyperparameters for training and testing."""

    # Experiment parameters
    checkpoint_dir: Path = field(
        default=Path("models/checkpoints"),
        metadata={"help": "Directory to save checkpoints"},
    )
    data_dir: Path = field(
        default=Path("data/train"), metadata={"help": "Directory to save data"}
    )
    val_data_dir: Path = field(
        default=Path("data/valid"),
        metadata={"help": "Directory to save validation data"},
    )
    load_checkpoint_step: int | None = field(
        default=None,
        metadata={
            "help": "Step to load the checkpoint from. If None, checkpoint is not loaded"
        },
    )
    log_dir: Path = field(
        default=Path("models/logs"),
        metadata={"help": "Directory to save tensorboard logs"},
    )
    video_dir: Path = field(
        default=Path("models/videos"), metadata={"help": "Directory to save videos"}
    )
    experiment_name: str = field(
        default="blokus", metadata={"help": "Name of the experiment"}
    )
    cuda: bool = field(default=True, metadata={"help": "Whether to use CUDA"})
    seed: int = field(default=42, metadata={"help": "Seed for the experiment"})
    wanda: bool = field(default=False, metadata={"help": "Whether to use wandb"})
    wandb_project_name: str = field(
        default="blokus", metadata={"help": "Wandb project name"}
    )
    capture_video: bool = field(
        default=False, metadata={"help": "Whether to capture video"}
    )
    log_interval: int = field(
        default=5, metadata={"help": "Number of updates between logging"}
    )
    tb_log_interval: int = field(
        default=10,
        metadata={"help": "Number of updates between logging to tensorboard"},
    )
    logging: bool = field(
        default=True, metadata={"help": "Whether to log to stdout and tensorboard"}
    )
    detect_anomaly: bool = field(
        default=False, metadata={"help": "Whether to detect autograd anomalies"}
    )
    verbose: bool = field(
        default=False, metadata={"help": "Whether to print debug information"}
    )

    def __post_init__(self):
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.experiment_name}_{self.start_time}"
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.video_dir = Path(self.video_dir)
        self.log_dir = Path(self.log_dir)
        self.data_dir = Path(self.data_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def dump_to_yaml(self, hparam_fp: Path) -> None:
        """Save hyperparameters to a YAML file."""
        with open(hparam_fp, "w", encoding="utf-8") as f:
            yaml.dump(self._hparams_to_dict(), f, default_flow_style=False)

    def _hparams_to_dict(self):
        """Convert hyperparameters to a dictionary."""
        hparams_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                hparams_dict[key] = str(value)
            else:
                hparams_dict[key] = value
        return hparams_dict


@dataclass
class PPOHparams(HParams):
    """Hyperparameters for PPO."""

    # Environment parameters
    gym_env: str = field(
        default="blokus_gym:blokus-simple-v0", metadata={"help": "Gym environment ID"}
    )
    num_envs: int = field(
        default=4, metadata={"help": "Number of parallel environments"}
    )

    # Model parameters
    agent_type: Literal["mlp", "cnn"] = field(
        default="mlp", metadata={"help": "Type of agent to use"}
    )
    save_interval: int = field(
        default=100, metadata={"help": "Number of updates between saving checkpoints"}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout probability for MLP"}
    )
    d_model: int = field(default=64, metadata={"help": "Dimensionality of the model"})

    # CNN parameters
    cnn_layers: int = field(default=4, metadata={"help": "Number of layers for CNN"})
    cnn_channels: int = field(
        default=1, metadata={"help": "Number of channels for CNN"}
    )
    cnn_kernel_size: int = field(default=3, metadata={"help": "Kernel size for CNN"})
    cnn_stride: int = field(default=1, metadata={"help": "Stride for CNN"})
    cnn_padding: int = field(default=1, metadata={"help": "Padding for CNN"})
    cnn_dropout: float = field(
        default=0.1, metadata={"help": "Dropout probability for CNN"}
    )

    # Training parameters
    update_epochs: int = field(
        default=4, metadata={"help": "Number of epochs to update the network"}
    )
    learning_rate: float = field(default=2.5e-4, metadata={"help": "Learning rate"})
    total_timesteps: int = field(
        default=500_000, metadata={"help": "Total number of timesteps"}
    )
    num_steps: int = field(
        default=128, metadata={"help": "Number of steps to run for each environment"}
    )
    num_minibatches: int = field(
        default=4, metadata={"help": "Number of minibatches to split the batch into"}
    )
    eps: float = field(default=1e-5, metadata={"help": "Adam epsilon"})
    anneal_lr: bool = field(
        default=True, metadata={"help": "Whether to anneal the learning rate"}
    )
    gae: bool = field(
        default=True,
        metadata={"help": "Whether to use generalized advantage estimation"},
    )
    gamma: float = field(default=0.99, metadata={"help": "Discount factor for rewards"})
    gae_lambda: float = field(
        default=0.95, metadata={"help": "Lambda parameter for GAE"}
    )
    clip_coef: float = field(
        default=0.2, metadata={"help": "Clipping parameter for PPO"}
    )
    norm_adv: bool = field(
        default=True, metadata={"help": "Whether to normalize the advantages"}
    )
    clip_vloss: bool = field(
        default=True, metadata={"help": "Whether to clip the value loss"}
    )
    ent_coef: float = field(
        default=0.01, metadata={"help": "Entropy coefficient for PPO"}
    )
    vf_coef: float = field(
        default=0.5, metadata={"help": "Value function coefficient for PPO"}
    )
    max_grad_norm: float = field(
        default=0.5, metadata={"help": "Maximum gradient norm for gradient clipping"}
    )
    target_kl: float = field(
        default=0.01, metadata={"help": "Target KL divergence for PPO"}
    )

    def __post_init__(self):
        super().__post_init__()
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_updates = self.total_timesteps // self.batch_size


@dataclass
class AlphaZeroHparams(HParams):
    """Hyperparameters for MCTS."""

    # Board parameters
    board_size: int = field(default=20, metadata={"help": "Size of the board"})
    number_of_players: int = field(
        default=4, metadata={"help": "Number of players in the game"}
    )
    states_dir: Path = field(
        default=Path("states"), metadata={"help": "Dir for blokus game states"}
    )

    # Model parameters
    model_type: Literal["dumbnet", "dcnnet", "resnet"] = field(
        default="resnet", metadata={"help": "Type of model to use"}
    )
    num_res_blocks: int = field(
        default=5, metadata={"help": "Number of residual blocks"}
    )
    lr: float = field(default=0.001, metadata={"help": "Learning rate"})
    dropout: float = field(default=0.3, metadata={"help": "Dropout probability"})
    epochs: int = field(default=10, metadata={"help": "Number of epochs"})
    batch_size: int = field(default=64, metadata={"help": "Batch size"})
    num_channels: int = field(default=128, metadata={"help": "Number of channels"})
    linear_dim: int = field(default=128, metadata={"help": "Linear layer dimension"})
    num_updates: int = field(default=1000, metadata={"help": "Number of updates"})
    weight_decay: float = field(default=1e-4, metadata={"help": "Weight decay"})

    # Training parameters
    num_iters: int = field(default=1000, metadata={"help": "Number of iterations"})
    num_eps: int = field(default=100, metadata={"help": "Number of episodes"})
    temp_threshold: int = field(default=15, metadata={"help": "Temperature threshold"})
    update_threshold: float = field(default=0.6, metadata={"help": "Update threshold"})
    max_len_of_queue: int = field(
        default=200000, metadata={"help": "Maximum length of queue"}
    )
    num_mcts_sims: int = field(
        default=100, metadata={"help": "Number of MCTS simulations"}
    )
    arena_num_mcts_sims: int = field(
        default=50,
        metadata={"help": "Number of MCTS simulations played during arena comparision"},
    )
    compare_arena_games: int = field(
        default=24, metadata={"help": "Number of games to play in the compare arena"}
    )
    permute: bool = field(
        default=True,
        metadata={"help": "Whether to permute players in the compare arena"},
    )
    cpuct: int = field(default=1, metadata={"help": "CPUCT"})
    elo_convert_rate: int = field(
        default=20, metadata={"help": "Elo convert rate for elo calculation"}
    )
    buffer_size_limit: int = field(
        default=10000, metadata={"help": "Buffer size limit"}
    )
    temperature: float = field(default=1.0, metadata={"help": "Temperature"})

    # Checkpoint parameters
    best_model_name: str = field(
        default="best.pth.tar", metadata={"help": "Name of the best model"}
    )
    temp_model_name: str = field(
        default="temp.pth.tar", metadata={"help": "Name of the temporary model"}
    )
    num_iters_for_train_examples_history: int = field(
        default=20, metadata={"help": "Number of iterations for train examples history"}
    )
    skip_first_self_play: bool = field(
        default=False, metadata={"help": "Whether to skip the first self play"}
    )

    # Evaluation parameters
    arena_players: list[str] = field(
        default_factory=lambda: ["checkpoint", "random", "uninformed"],
        metadata={"help": "Players to use in the arena"},
    )
    opponent_strength: int = field(
        default=20, metadata={"help": "Strength of the opponent"}
    )
    opponent_type: Literal["pnet", "random", "uninformed"] = field(
        default="uninformed",
        metadata={"help": "Type of the opponent (random or checkpoint)"},
    )


HParamsType = HParams | PPOHparams | AlphaZeroHparams


def get_hparams(algorithm: str = "ppo"):
    """Get hyperparameters for a given algorithm."""
    hparams_dict = {"ppo": PPOHparams, "alphazero": AlphaZeroHparams}
    return hparams_dict[algorithm]


def load_hparams(hparam_fp: Path | str | None = None, algorithm: str = "ppo"):
    """Load hyperparameters from a YAML file."""
    if hparam_fp is None:
        return get_hparams(algorithm)
    with open(hparam_fp, "r", encoding="utf-8") as f:
        hparams_dict = yaml.safe_load(f)
    return get_hparams(algorithm)(**hparams_dict)
