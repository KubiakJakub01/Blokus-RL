"""Module containing the HumanPlayer class for playing blokus as a human."""
import numpy as np

from .player import Player
from ..utils import log_info


class HumanPlayer(Player):
    """Class for playing blokus as a human."""

    def __init__(self, game):
        self.game = game

    def update_state(self, s, current_player):
        """Update the state of the player.

        Args:
            s: The current state.
            current_player: The current player.

        Returns:
            The updated state and current player.
        """
        log_info(s)
        a = input("Enter move: ")
        a = np.array([int(a)])
        s_prime, current_player = self.game.get_next_state(
            s, current_player, a[0]
        )
        return s_prime, current_player

    def reset(self):
        """Reset the player."""
        pass
