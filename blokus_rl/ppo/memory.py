"""Memory for PPO."""
import gymnasium as gym
import torch

from ..hparams import HParams


class Memory:
    """Memory for PPO."""

    def __init__(self, hparams: HParams, envs: gym.envs, device: torch.device):
        """Initialize the memory."""
        self.num_steps = hparams.num_steps
        self.num_envs = hparams.num_envs
        self.obs = torch.zeros(
            (self.num_steps, self.num_envs) + envs.single_observation_space.shape
        ).to(device)
        self.actions = torch.zeros(
            (self.num_steps, self.num_envs) + envs.single_action_space.shape
        ).to(device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.advantages = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.returns = torch.zeros((self.num_steps, self.num_envs)).to(device)
