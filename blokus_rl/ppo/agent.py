"""Deep neural network model for PPO algorithm."""
import numpy as np
import torch
from torch import nn
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


class FilterLegalMoves(nn.Module):
    """Filter out illegal moves."""

    def __init__(self):
        super().__init__()

    def forward(self, x, possible_moves):
        actions_tensor = torch.zeros(x.shape).to(x.device)
        # Create a mask to filter out illegal moves
        mask = torch.zeros(x.shape).to(x.device)
        for i, possible_move in enumerate(possible_moves):
            mask[i, possible_move] = 1
        # Apply mask
        actions_tensor = x * mask
        actions_tensor[actions_tensor == 0] = -1e9
        return actions_tensor


class ConvBlock(nn.Module):
    """Convolutional neural networks block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 4,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1,
    ):
        """Convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_layers: Number of convolutional layers.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
            dropout: Dropout probability.
            std: Standard deviation of the weights.

        Returns:
            Initialized convolutional block."""
        super().__init__()
        assert n_layers >= 1, "Number of layers must be at least 1"
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        )
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv_block(x)
        return x


class CnnAgent(nn.Module):
    """Agent with a convolutional block."""

    def __init__(self, envs, hparams: HParams):
        """Initialize the agent.

        Args:
            envs: Environment object.
            hparams: Hyperparameters."""
        super().__init__()
        self.board_dim = np.array(envs.single_observation_space.shape).prod()
        self.output_dim = envs.single_action_space.n
        self.d_model = hparams.d_model
        self.conv_block = ConvBlock(
            1,
            self.d_model,
            hparams.cnn_layers,
            hparams.cnn_kernel_size,
            hparams.cnn_stride,
            hparams.cnn_padding,
            hparams.cnn_dropout,
        )
        self.actor = MLP(
            self.d_model * self.board_dim,
            self.d_model,
            self.output_dim,
            hparams.dropout,
            std=0.01,
        )
        self.critic = MLP(
            self.d_model * self.board_dim, self.d_model, 1, hparams.dropout, std=1.0
        )

    def forward(self, x, possible_moves=None):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.actor(x, possible_moves)
        return x

    def get_value(self, x):
        """Get the value of a state."""
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the conv block
        return self.critic(x)

    def get_action_and_value(self, x, action=None, possible_moves=None):
        """Get an action and its value."""
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the conv block
        logits = self.actor(x, possible_moves)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


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
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Dropout(dropout),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim * 2)),
            nn.Dropout(dropout),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.Dropout(dropout),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, output_dim), std),
        )
        self.filter_legal_moves = FilterLegalMoves()

    def forward(self, x, possible_moves=None):
        x = self.net(x)
        if possible_moves is not None:
            x = self.filter_legal_moves(x, possible_moves)
        return x


class Agent(nn.Module):
    """Agent network."""

    def __init__(self, envs, hparams: HParams):
        """Initialize the agent.

        Args:
            envs: Environment object.
            hparams: Hyperparameters.
        """
        super().__init__()
        self.input_dim = np.array(envs.single_observation_space.shape).prod()
        self.output_dim = envs.single_action_space.n
        self.d_model = hparams.d_model

        self.actor = MLP(
            self.input_dim, self.d_model, self.output_dim, hparams.dropout, std=0.01
        )
        self.critic = MLP(self.input_dim, self.d_model, 1, hparams.dropout, std=1.0)

    def forward(self, x, possible_moves=None):
        x = self._preprocess(x)
        x = self.actor(x, possible_moves)
        return x

    def get_value(self, x):
        """Get the value of a state."""
        x = self._preprocess(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None, possible_moves=None):
        """Get an action and its value."""
        x = self._preprocess(x)
        logits = self.actor(x, possible_moves)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def _preprocess(self, x):
        """Preprocess the input."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim >= 3:
            x = x.reshape(-1, self.input_dim)
        return x


def get_agent(agent: str):
    agents_dict = {"mlp": Agent, "cnn": CnnAgent}
    return agents_dict[agent]
