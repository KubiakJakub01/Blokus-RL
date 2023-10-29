"""Module with the MCTSPlayer class for playing games using MCTS."""
import numpy as np

from ..mcts import MCTS
from ..neural_network import BlokusNNetWrapper
from .player import Player


class MCTSPlayer(Player):
    def __init__(self, game, nn, simulations):
        self.game = game
        self.simulations = simulations
        self.nn = nn
        self.tree = MCTS(game, nn)

    def update_state(self, s, current_player, terminal, winners):
        for _ in range(self.simulations):
            self.tree.simulate(s, current_player, terminal, winners)

        dist = self.tree.get_distribution(s, 0)
        a = dist[np.argmax(dist[:, 1]), 0]
        s_prime, current_player, terminal, winners = self.game.get_next_state(
            s, current_player, a[0]
        )
        return s_prime, current_player, terminal, winners

    def reset(self):
        self.tree = MCTS(self.game, self.nn)
