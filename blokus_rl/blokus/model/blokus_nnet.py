"""Deep neural network model for PPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ...hparams import MCTSHparams
from ...utils import AverageMeter, LOG_INFO
from ..blokus_wrapper import BlokusGameWrapper


class FilterLegalMoves(nn.Module):
    """Filter out illegal moves."""

    def __init__(self):
        super(FilterLegalMoves, self).__init__()

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


class BlokusNNet(nn.Module):
    def __init__(self, game: BlokusGameWrapper, hparams: MCTSHparams):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.hparams = hparams

        super(BlokusNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, hparams.num_channels, 3, stride=1, padding=1)
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

        self.fc3 = nn.Linear(self.hparams.linear_dim // 2, self.action_size)

        self.fc4 = nn.Linear(self.hparams.linear_dim // 2, 1)

    def forward(self, s):
        # s: batch_size x board_x x board_y
        s = s.view(
            -1, 1, self.board_x, self.board_y
        )  # batch_size x 1 x board_x x board_y
        s = F.relu(
            self.bn1(self.conv1(s))
        )  # batch_size x num_channels x board_x x board_y
        s = F.relu(
            self.bn2(self.conv2(s))
        )  # batch_size x num_channels x board_x x board_y
        s = F.relu(
            self.bn3(self.conv3(s))
        )  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(
            self.bn4(self.conv4(s))
        )  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(
            -1, self.hparams.num_channels * (self.board_x - 4) * (self.board_y - 4)
        )

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.hparams.dropout,
            training=self.training,
        )  # batch_size x 1024
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.hparams.dropout,
            training=self.training,
        )  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class BlokusNNetWrapper:
    def __init__(
        self, game: BlokusGameWrapper, hparams: MCTSHparams, device: str = "cpu"
    ):
        self.hparams = hparams
        self.device = device
        self.nnet = BlokusNNet(game, hparams).to(self.device)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.elo = 1000

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(
            self.nnet.parameters(),
            lr=self.hparams.lr,
        )
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        total_losses = AverageMeter()

        with tqdm(range(self.hparams.epochs), desc="Training Net...") as t:
            for epoch in t:
                t.set_description("Training Net (epoch #{})".format(epoch + 1))
                self.nnet.train()

                batch_count = int(len(examples) / self.hparams.batch_size)

                for _ in range(batch_count):
                    sample_ids = np.random.randint(
                        len(examples), size=self.hparams.batch_size
                    )
                    boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                    boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(
                        self.device
                    )
                    target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(
                        self.device
                    )

                    # compute output
                    out_pi, out_v = self.nnet(boards)
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v

                    # record loss
                    pi_losses.update(l_pi.item(), boards.size(0))
                    v_losses.update(l_v.item(), boards.size(0))
                    total_losses.update(total_loss.item(), boards.size(0))
                    t.set_postfix(
                        Loss_pi=pi_losses, Loss_v=v_losses, Total_loss=total_losses
                    )

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

        return pi_losses, v_losses, total_losses

    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        board = torch.Tensor(board.astype(np.float64)).to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, filename: str = "checkpoint.pth.tar"):
        """Save the model."""
        model_path = self.hparams.checkpoint_dir / filename
        LOG_INFO("Saving checkpoint to: %s", model_path)
        torch.save({"nnet": self.nnet.state_dict(), "elo": self.elo}, model_path)

    def load_checkpoint(self, filename: str = "checkpoint.pth.tar"):
        """Load the model."""
        model_path = self.hparams.checkpoint_dir / filename
        LOG_INFO("Loading model from: %s", str(model_path))
        assert model_path.exists(), f"Model path doesn't exist {model_path}"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.nnet.load_state_dict(checkpoint["nnet"])
        self.elo = checkpoint["elo"]
