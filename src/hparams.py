"""Hyperparameters for training and testing."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class HParams:
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
    seed: int = field(default=42, metadata={"help": "Seed for the experiment"})
    num_envs: int = field(
        default=4, metadata={"help": "Number of parallel environments"}
    )
    gym_env: str = field(
        default="blokus_gym:blokus-simple-v0", metadata={"help": "Gym environment ID"}
    )
    wanda: bool = field(default=False, metadata={"help": "Whether to use wandb"})
    wandb_project_name: str = field(
        default="blokus", metadata={"help": "Wandb project name"}
    )
    wandb_entity: str = field(default="blokus", metadata={"help": "Wandb entity"})
    capture_video: bool = field(
        default=False, metadata={"help": "Whether to capture video"}
    )
    num_epochs: int = field(
        default=100, metadata={"help": "Number of epochs to train for"}
    )
    learning_rate: float = field(default=2.5e-4, metadata={"help": "Learning rate"})
    total_timesteps: int = field(
        default=1_000_000, metadata={"help": "Total number of timesteps"}
    )
    num_steps: int = field(
        default=128, metadata={"help": "Number of steps to run for each environment"}
    )
    batch_size: int = field(default=256, metadata={"help": "Batch size"})
    minibatch_size: int = field(
        default=32, metadata={"help": "Minibatch size for gradient descent"}
    )
    eps: float = field(default=1e-5, metadata={"help": "Adam epsilon"})
    anneal_lr: bool = field(
        default=True, metadata={"help": "Whether to anneal the learning rate"}
    )
    gae: bool = field(
        default=True, metadata={"help": "Whether to use generalized advantage estimation"}
    )
    gamma: float = field(
        default=0.99, metadata={"help": "Discount factor for rewards"}
    )
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

    def __post_init__(self):
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.experiment_name}_{self.start_time}"
        self.num_updates = self.total_timesteps // self.batch_size


def load_params(hparam_fp: Path | None = None) -> HParams:
    """Load hyperparameters from a YAML file."""
    if hparam_fp is None:
        return HParams()
    with open(hparam_fp, "r", encoding="utf-8") as f:
        hparams_dict = yaml.safe_load(f)
    return HParams(**hparams_dict)
