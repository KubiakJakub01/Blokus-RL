"""Hyperparameters for training and testing."""
import os
from dataclasses import dataclass, field


@dataclass
class HParams:
    experiment_name: str = field(default="blokus", type=str, metadata={"help": "Name of the experiment"})
    seed: int = field(default=42, type=int, metadata={"help": "Seed for the experiment"})
    num_envs: int = field(default=4, type=int, metadata={"help": "Number of parallel environments"})
    gym_env: str = field(default="blokus_gym:blokus-simple-v0", type=str, metadata={"help": "Gym environment ID"})
