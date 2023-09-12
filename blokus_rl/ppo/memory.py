"""Memory for PPO."""
import gymnasium as gym
import torch

from ..hparams import HParams


class Memory:
    """Memory for PPO."""

    def __init__(self, hparams: HParams, envs: gym.envs, device: torch.device):
        """Initialize the memory."""
        self.hparams = hparams
        self.envs = envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs = torch.zeros(
            (self.hparams.num_steps, self.hparams.num_envs)
            + self.envs.single_observation_space.shape
        ).to(self.device)
        self.actions = torch.zeros(
            (self.hparams.num_steps, self.hparams.num_envs)
            + self.envs.single_action_space.shape
        ).to(self.device)
        self.logprobs = torch.zeros((self.hparams.num_steps, self.hparams.num_envs)).to(
            self.device
        )
        self.rewards = torch.zeros((self.hparams.num_steps, self.hparams.num_envs)).to(
            self.device
        )
        self.dones = torch.zeros((self.hparams.num_steps, self.hparams.num_envs)).to(
            self.device
        )
        self.values = torch.zeros((self.hparams.num_steps, self.hparams.num_envs)).to(
            self.device
        )
