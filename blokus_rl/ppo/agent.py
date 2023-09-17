"""Deep neural network model for PPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from ..hparams import HParams


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    """Initialize a linear layer with orthogonal initialization.
    
    Args:
        layer: Linear layer to initialize.
        std: Standard deviation of the weights.
        bias_const: Constant to initialize the bias with.
    
    Returns:
        Initialized linear layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, std=0.01):
        """Multi-layer perceptron.
        
        Args:
            input_dim: Dimensionality of the input.
            hidden_dim: Dimensionality of the hidden layers.
            output_dim: Dimensionality of the output.
            dropout: Dropout probability.
            std: Standard deviation of the weights.
            
        Returns:
            Initialized MLP."""
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Dropout(dropout),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.Dropout(dropout),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim // 2, hidden_dim // 8)),
            nn.Dropout(dropout),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim // 8, output_dim), std),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Agent(nn.Module):
    """Agent network."""
    def __init__(self, envs, hparams: HParams):
        """Initialize the agent.

        Args:
            envs: Environment object.
            hparams: Hyperparameters.
        """
        super(Agent, self).__init__()
        self.input_dim = np.array(envs.single_observation_space.shape).prod()
        self.output_dim = envs.single_action_space.n
        self.d_model = hparams.d_model

        self.actor = MLP(self.input_dim, self.d_model, self.output_dim, hparams.dropout, std=0.01)
        self.critic = MLP(self.input_dim, self.d_model, 1, hparams.dropout, std=1.0)

    def get_value(self, x):
        """Get the value of a state."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Get an action and its value."""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
