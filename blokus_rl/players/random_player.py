"""Module with random player class."""
from .player import Player


class RandomPlayer(Player):
    """Player that selects actions uniformly at random."""

    def __init__(self, game):
        self.game = game

    def update_state(self, s, current_player):
        """Update the state of the player."""
        a = self.game.get_sample_move(s)
        s_prime, current_player = self.game.get_next_state(
            s, current_player, a
        )
        return s_prime, current_player

    def reset(self):
        """Reset the player."""
        return
