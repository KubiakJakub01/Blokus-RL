"""Module with trainer for class."""
import os
from pathlib import Path

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.hparams import HParams
from src.utils import LOG_INFO, make_env


class PPOAgent:
    """Trainer for proximal policy optimization."""
    def __init__(self, hparams: HParams):
        """Initialize the trainer."""
        self.hparams = hparams
        self.envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    hparams.gym_env,
                    hparams.seed,
                    i,
                    hparams.capture_video,
                    hparams.run_name,
                )
                for i in range(hparams.num_envs)
            ]
        )
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"

    def train(self):
        """Train the model."""
