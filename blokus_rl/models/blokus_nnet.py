import torch
import torch.nn.functional as F
from torch import nn

from ..colossumrl import ColosseumBlokusGameWrapper
from ..hparams import MCTSHparams


class DCNNet(nn.Module):
    """Deep Convolutional Neural Network for Blokus"""

    def __init__(self, game: ColosseumBlokusGameWrapper, hparams: MCTSHparams):
        super().__init__()
        # game params
        self.hparams = hparams
        self.board_x, self.board_y = game.get_board_size()  # 20, 20
        self.action_size = game.get_action_size()  # 30433
        self.num_players = game.number_of_players  # 4

        # Neural Net
        # Input have shape (batch_size, players*2, board_x, board_y)
        # Output have shape (batch_size, action_size)
        self.conv1 = nn.Conv2d(
            self.num_players * 2, hparams.num_channels, 3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            hparams.num_channels, hparams.num_channels, 3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(hparams.num_channels, hparams.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(hparams.num_channels, hparams.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(hparams.num_channels)
        self.bn2 = nn.BatchNorm2d(hparams.num_channels)
        self.bn3 = nn.BatchNorm2d(hparams.num_channels)
        self.bn4 = nn.BatchNorm2d(hparams.num_channels)

        self.fc1 = nn.Linear(
            hparams.num_channels * (self.board_x - 4) * (self.board_y - 4),
            self.hparams.linear_dim,
        )
        self.fc_bn1 = nn.BatchNorm1d(self.hparams.linear_dim)

        self.fc2 = nn.Linear(self.hparams.linear_dim, self.hparams.linear_dim // 2)
        self.fc_bn2 = nn.BatchNorm1d(self.hparams.linear_dim // 2)

        # This is the policy head
        self.fc3 = nn.Linear(self.hparams.linear_dim // 2, self.action_size)

        # This is the value head
        self.fc4 = nn.Linear(self.hparams.linear_dim // 2, self.num_players)

    def forward(self, x):
        # n_boards = n_player*2
        # s: batch_size x n_boards x board_x x board_y
        x = F.relu(
            self.bn1(self.conv1(x))
        )  # batch_size x num_channels x board_x x board_y
        x = F.relu(
            self.bn2(self.conv2(x))
        )  # batch_size x num_channels x board_x x board_y
        x = F.relu(
            self.bn3(self.conv3(x))
        )  # batch_size x num_channels x (board_x-2) x (board_y-2)
        x = F.relu(
            self.bn4(self.conv4(x))
        )  # batch_size x num_channels x (board_x-4) x (board_y-4)
        x = x.view(
            -1, self.hparams.num_channels * (self.board_x - 4) * (self.board_y - 4)
        )

        x = F.dropout(
            F.relu(self.fc_bn1(self.fc1(x))),
            p=self.hparams.dropout,
            training=self.training,
        )  # batch_size x 1024
        x = F.dropout(
            F.relu(self.fc_bn2(self.fc2(x))),
            p=self.hparams.dropout,
            training=self.training,
        )  # batch_size x 512

        pi = self.fc3(x)  # batch_size x action_size
        v = self.fc4(x)  # batch_size x 4

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class ResNet(nn.Module):
    """Residual Neural Network for Blokus"""

    def __init__(self, game: ColosseumBlokusGameWrapper, hparams: MCTSHparams):
        super().__init__()

        # game params
        self.hparams = hparams
        self.board_x, self.board_y = game.get_board_size()  # 20, 20
        self.action_size = game.get_action_size()  # 30433
        self.num_players = game.number_of_players  # 4
        self.input_dim = game.get_observation_size()  # (8, 20, 20)

        self.conv1 = nn.Conv2d(self.input_dim[0], 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # A series of residual blocks
        self.res_blocks = nn.Sequential(
            *[self._build_res_block(64) for _ in range(self.hparams.num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)  # 2 filters for policy head
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_out = nn.Linear(
            2 * self.input_dim[1] * self.input_dim[2], self.action_size
        )

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1 filter for value head
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.input_dim[1] * self.input_dim[2], 64)
        self.value_fc2 = nn.Linear(64, self.num_players)

    def _build_res_block(self, channel_in):
        block = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_in),
            nn.ReLU(),
            nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_in),
        )
        return block

    def _residual(self, x, residual_function):
        return F.relu(x + residual_function(x))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self._residual(x, self.res_blocks)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_out(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
