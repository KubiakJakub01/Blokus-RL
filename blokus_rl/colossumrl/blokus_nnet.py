import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from ..hparams import MCTSHparams
from ..utils import LOG_INFO, AverageMeter, to_device
from .blokus_wrapper import ColosseumBlokusGameWrapper


class BlokusNNet(nn.Module):
    def __init__(self, game: ColosseumBlokusGameWrapper, hparams: MCTSHparams):
        super(BlokusNNet, self).__init__()
        # game params
        self.hparams = hparams
        self.board_x, self.board_y = game.get_board_size() # 20, 20
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
        v = self.fc4(x)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


# Object that manages interfacing data with the underlying PyTorch model, as well as checkpointing models.
class BlokusNNetWrapper():

    def __init__(self, game: ColosseumBlokusGameWrapper, hparams: MCTSHparams, device: str = "cpu"):
        self.game = game
        self.hparams = hparams
        self.device = device
        self.model = BlokusNNet(game, hparams).to(self.device)
        self.elo = 1000
        self.latest_loss = 0
        self.mean_loss = AverageMeter()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay)

    def train_step(self, batch):
        """Train the model for one step.
        
        Args:
            batch: A batch of data.
        
        Returns:
            The loss."""
        self.model.train()

        batch = to_device(batch, self.device)
        obs = batch['observation']

        p_pred, v_pred = self.model(obs)

        masks = batch['mask']
        p_gt = batch['prob']
        v_gt = batch['score']

        loss = self.loss(masks, (p_pred, v_pred), (p_gt, v_gt))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.latest_loss = loss.item()
        self.mean_loss.update(self.latest_loss)

        return loss.item()

    def predict(self, x, mask):
        """Predict the policy and value for a given state.

        Args:
            x: The state.
            mask: The mask of valid actions.

        Returns:
            The policy and value."""
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(0)
            x = x.to(self.device)
            p_logits, v = self.model(x)
            mask = torch.from_numpy(mask).bool().to(self.device)
            p, v = self.get_valid_dist(mask, p_logits[0]).cpu().numpy().squeeze(), v.cpu().numpy().squeeze() # EXP because log softmax
        return p, v

    def loss(self, masks, prediction, target):
        """Compute the loss.
        
        Args:
            masks: The mask of valid actions.
            prediction: The prediction.
            target: The target.
        
        Returns:
            The loss."""
        p_pred, v_pred = prediction
        p_gt, v_gt = target
        v_loss = ((v_pred - v_gt)**2).sum()  # Mean squared error
        p_loss = 0
        # TODO: Make sure if loop here is necessary
        for mask, gt, logits in zip(masks, p_gt, p_pred):
            pred = self.get_valid_dist(mask, logits, log_softmax=True)
            p_loss += -torch.sum(gt * pred)
        return p_loss + v_loss

    def get_valid_dist(self, mask, logits, log_softmax=False):
        """Get the valid distribution.

        Args:
            mask: The mask of valid actions.
            logits: The logits.
            log_softmax: Whether to return the log softmax.
        
        Returns:
            The valid distribution."""
        selection = torch.masked_select(logits, mask)
        dist = torch.nn.functional.log_softmax(selection, dim=-1)
        if log_softmax:
            return dist
        return torch.exp(dist)

    def save_checkpoint(self, filename: str = "checkpoint.pth.tar"):
        """Save the model."""
        model_path = self.hparams.checkpoint_dir / filename
        LOG_INFO("Saving checkpoint to: %s", model_path)
        torch.save({"nnet": self.model.state_dict(), "elo": self.elo}, model_path)

    def load_checkpoint(self, filename: str = "checkpoint.pth.tar"):
        """Load the model."""
        model_path = self.hparams.checkpoint_dir / filename
        LOG_INFO("Loading model from: %s", str(model_path))
        assert model_path.exists(), f"Model path doesn't exist {model_path}"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["nnet"])
        self.elo = checkpoint["elo"]
