"""Hyperparameters for training and testing."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

@dataclass
class HParams:
    experiment_name: str = field(default="blokus", metadata={"help": "Name of the experiment"})
    seed: int = field(default=42, metadata={"help": "Seed for the experiment"})
    num_envs: int = field(default=4, metadata={"help": "Number of parallel environments"})
    gym_env: str = field(default="blokus_gym:blokus-simple-v0", metadata={"help": "Gym environment ID"})


def load_params(hparam_fp: Path | None = None) -> HParams:
    """Load hyperparameters from a YAML file."""
    if hparam_fp is None:
        return HParams()
    with open(hparam_fp, "r", encoding="utf-8") as f:
        hparams_dict = yaml.safe_load(f)
    return HParams(**hparams_dict)
