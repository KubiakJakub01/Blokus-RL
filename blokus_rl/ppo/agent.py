"""Deep neural network model for PPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from ..hparams import HParams


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Critic(nn.Module):
    def __init__(self, envs, hparams: HParams):
        super(Critic, self).__init__()
        self.d_model = hparams.d_model
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.d_model)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(self.d_model, self.d_model)),
            nn.Tanh(),
            layer_init(nn.Linear(self.d_model, 1), std=1.0),
        )

    def forward(self, x):
        return self.critic(x)


class Actor(nn.Module):
    def __init__(self, envs, hparams: HParams):
        super(Actor, self).__init__()
        self.d_model = hparams.d_model
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.d_model)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(self.d_model, self.d_model)),
            nn.Tanh(),
            layer_init(nn.Linear(self.d_model, envs.single_action_space.n), std=0.01),
        )

    def forward(self, x):
        return self.actor(x)


class Agent(nn.Module):
    def __init__(self, envs, hparams: HParams):
        super(Agent, self).__init__()
        self.critic = Critic(envs, hparams)
        self.actor = Actor(envs, hparams)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
