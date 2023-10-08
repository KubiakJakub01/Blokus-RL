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
        self.advantages = torch.zeros(
            (self.hparams.num_steps, self.hparams.num_envs)
        ).to(self.device)
        self.returns = torch.zeros((self.hparams.num_steps, self.hparams.num_envs)).to(
            self.device
        )

    def get_flatten_batch(self):
        """Return a flattened batch of data."""
        return {
            "obs": self.obs.reshape((-1,) + self.envs.single_observation_space.shape),
            "logprobs": self.logprobs.reshape(-1),
            "actions": self.actions.reshape(
                (-1,) + self.envs.single_action_space.shape
            ),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
        }
