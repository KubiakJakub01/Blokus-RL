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
from src.ppo.agent import Agent


class Trainer:
    """Trainer for proximal policy optimization."""

    def __init__(self, hparams: HParams, device: torch.device):
        """Initialize the trainer."""
        self.hparams = hparams
        self.device = device
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(i, hparams) for i in range(hparams.num_envs)]
        )
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"

        # Init variables
        self.global_step = 0
        self.next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        self.next_done = torch.zeros(hparams.num_envs).to(self.device)

        # Init model
        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=hparams.lr, eps=hparams.eps
        )

        # Init tensorboard
        self.writer = SummaryWriter(log_dir=f"{hparams.log_dir}/{hparams.run_name}")

    def train(self):
        """Train the model."""
