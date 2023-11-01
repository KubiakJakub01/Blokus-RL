"""Module with random player class."""
import numpy as np

from .player import Player


class RandomPlayer(Player):
    """Player that selects actions uniformly at random."""

    def __init__(self, game):
        self.game = game

    def update_state(self, s, current_player, terminal, winners):
        """Update the state of the player."""
        a = np.random.choice(self.game.get_legal_actions(s, current_player))
        s_prime, current_player, terminal, winners = self.game.get_next_state(
            s, current_player, a
        )
        return s_prime, current_player, terminal, winners

    def reset(self):
        """Reset the player."""
        pass
