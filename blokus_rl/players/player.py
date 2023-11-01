"""Module defining the interface Player class."""
from abc import ABC, abstractmethod


class Player(ABC):
    @abstractmethod
    def update_state(self, s, current_player):
        """Update the state of the player."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the player."""
        raise NotImplementedError
