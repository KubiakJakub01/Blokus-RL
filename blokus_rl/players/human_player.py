"""Module containing the HumanPlayer class for playing blokus as a human."""
from .player import Player


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
        available_actions = self.game.get_valid_actions_for_human_player(s, current_player)
        self.game.display(s)
        for i, action in enumerate(available_actions):
            print(f'{i}: {action}')
        is_action_correct = False
        while not is_action_correct:
            a = input("Enter move: ")
            if a.isdigit() and int(a) in range(len(available_actions)):
                is_action_correct = True
            else:
                print("Invalid move. Try again.")
        action = available_actions[int(a)]
        print(f"You chose {action}")
        s_prime, current_player = self.game.get_next_state(
            s, current_player, action
        )
        return s_prime, current_player

    def reset(self):
        """Reset the player."""
        return

    def __str__(self) -> str:
        return "HumanPlayer"
