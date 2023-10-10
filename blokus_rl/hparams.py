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
    gym_env: str = field(
        default="blokus_gym:blokus-simple-v0", metadata={"help": "Gym environment ID"}
    )
    checkpoint_dir: Path = field(
        default=Path("models/checkpoints"),
        metadata={"help": "Directory to save checkpoints"},
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
    num_envs: int = field(
        default=4, metadata={"help": "Number of parallel environments"}
    )
    wanda: bool = field(default=False, metadata={"help": "Whether to use wandb"})
    wandb_project_name: str = field(
        default="blokus", metadata={"help": "Wandb project name"}
    )
    capture_video: bool = field(
        default=False, metadata={"help": "Whether to capture video"}
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = field(
        default="INFO", metadata={"help": "Logging level for the experiment"}
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

    def save(self, hparam_fp: Path) -> None:
        """Save hyperparameters to a YAML file."""
        with open(hparam_fp, "w", encoding="utf-8") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


@dataclass
class PPOHparams(HParams):
    """Hyperparameters for PPO."""
    # Model parameters
    agent_type: Literal["mlp", "cnn"] = field(
        default="mlp", metadata={"help": "Type of agent to use"}
    )
    save_interval: int = field(
        default=1000, metadata={"help": "Number of updates between saving checkpoints"}
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
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.experiment_name}_{self.start_time}"
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_updates = self.total_timesteps // self.batch_size

@dataclass
class HparamsMCTS:
    """Hyperparameters for MCTS."""
    # Model parameters
    lr: float = field(default=0.001, metadata={"help": "Learning rate"})
    dropout: float = field(default=0.3, metadata={"help": "Dropout probability"})
    epochs: int = field(default=10, metadata={"help": "Number of epochs"})
    batch_size: int = field(default=64, metadata={"help": "Batch size"})
    num_channels: int = field(default=128, metadata={"help": "Number of channels"})
    linear_dim: int = field(default=128, metadata={"help": "Linear layer dimension"})

    # Training parameters
    numIters: int = field(default=1000, metadata={"help": "Number of iterations"})
    numEps: int = field(default=100, metadata={"help": "Number of episodes"})
    tempThreshold: int = field(
        default=15, metadata={"help": "Temperature threshold"}
    )
    updateThreshold: float = field(
        default=0.6, metadata={"help": "Update threshold"}
    )
    maxlenOfQueue: int = field(
        default=200000, metadata={"help": "Maximum length of queue"}
    )
    numMCTSSims: int = field(
        default=25, metadata={"help": "Number of MCTS simulations"}
    )
    arenaCompare: int = field(
        default=40, metadata={"help": "Number of arena comparisons"}
    )
    cpuct: int = field(default=1, metadata={"help": "CPUCT"})

    # Checkpoint parameters
    load_model: bool = field(default=False, metadata={"help": "Whether to load model"})
    load_folder_file: tuple[str, str] = field(
        default=("./temp/", "best.pth.tar"), metadata={"help": "Folder to load model"}
    )
    numItersForTrainExamplesHistory: int = field(
        default=20, metadata={"help": "Number of iterations for train examples history"}
    )



def get_hparams(algorithm: str = "ppo") -> HParams:
    """Get hyperparameters for a given algorithm."""
    hparams_dict = {
        "ppo": PPOHparams,
    }
    return hparams_dict[algorithm]


def load_hparams(hparam_fp: Path | None = None, algorithm: str = "ppo") -> PPOHparams:
    """Load hyperparameters from a YAML file."""
    if hparam_fp is None:
        return get_hparams(algorithm)
    with open(hparam_fp, "r", encoding="utf-8") as f:
        hparams_dict = yaml.safe_load(f)
    return get_hparams(algorithm)(**hparams_dict)
