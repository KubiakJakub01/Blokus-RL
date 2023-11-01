"""DumbNet class."""
import torch
import torch.nn as nn


class DumbNet(nn.Module):
    """DumbNet class."""

    def __init__(self, game, hparams):
        super(DumbNet, self).__init__()
        self.p_shape = game.get_action_size()
        self.v_shape = game.get_number_of_players()

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]

        p_logits = torch.ones((batch_size, self.p_shape)).to(device)
        v = torch.zeros((batch_size, self.v_shape)).to(device)

        return p_logits, v
