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


class ToyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, std=0.01):
        super(ToyModel, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, output_dim), std),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, std=0.01):
        super().__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Dropout(dropout),
            nn.PReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Dropout(dropout),
            nn.PReLU(),
            layer_init(nn.Linear(hidden_dim, output_dim), std),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Agent(nn.Module):
    def __init__(self, envs, hparams: HParams):
        super(Agent, self).__init__()
        input_dim = np.array(envs.single_observation_space.shape).prod()
        output_dim = envs.single_action_space.n
        self.d_model = hparams.d_model

        if hparams.model_type == "mlp":
            self.actor = MLP(input_dim, self.d_model, output_dim, hparams.dropout, std=0.01)
            self.critic = MLP(input_dim, self.d_model, 1, hparams.dropout, std=1.0)
        elif hparams.model_type == "toy":
            self.actor = ToyModel(input_dim, self.d_model, output_dim, std=0.01)
            self.critic = ToyModel(input_dim, self.d_model, 1, std=1.0)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
