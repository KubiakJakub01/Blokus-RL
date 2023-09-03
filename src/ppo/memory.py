"""Memory for PPO."""
import torch


class Memory:
    """Memory for PPO."""

    def __init__(self, hparams, envs, device):
        """Initialize the memory."""
        self.num_steps = hparams.num_steps
        self.num_envs = hparams.num_envs
        obs = torch.zeros((self.num_steps, self.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((self.num_steps, self.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(device)
