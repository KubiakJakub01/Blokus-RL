"""Module with the MCTSPlayer class for playing games using MCTS."""
import numpy as np

from ..alphazero import MCTS
from .player import Player


class MCTSPlayer(Player):
    def __init__(self, game, nn, simulations):
        self.game = game
        self.simulations = simulations
        self.nn = nn
        self.tree = MCTS(game, nn)

    def update_state(self, s, current_player):
        for _ in range(self.simulations):
            self.tree.simulate(s, current_player)

        dist = self.tree.get_distribution(s, 0)
        a = dist[np.argmax(dist[:, 1]), 0]
        s_prime, current_player = self.game.get_next_state(s, current_player, a[0])
        return s_prime, current_player

    def reset(self):
        self.tree = MCTS(self.game, self.nn)

    def __str__(self) -> str:
        return "MCTSPlayer"
