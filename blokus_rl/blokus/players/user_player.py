# pylint: skip-file
import matplotlib.pyplot as plt

from ...hparams import MCTSHparams
from ..blokus_wrapper import BlokusGameWrapper
from ..game import blokus_game
from .player import Player


def human_play(game_wrapper: BlokusGameWrapper):
    """Human play loop."""
    game, player = game_wrapper.get_init_board()
    while game_wrapper.get_game_ended(game, player) == 0:
        game_wrapper.display(game, mode="tensor")
        print("Player ", str(player), ", it's your turn")
        possible_moves = {
            i: game_wrapper.all_possible_indexes_to_moves[i]
            for i in game.next_player().possible_move_indexes()
        }
        print("Possible moves:")
        for k, v in possible_moves.items():
            print(k, v)
        print("Enter move index: ")
        while True:
            try:
                index = int(input())
                if index not in possible_moves:
                    raise ValueError
                break
            except ValueError:
                print("Invalid move index, try again")
        game, player = game_wrapper.get_next_state(game, player, index)
    print(
        "Game over: Turn ",
        str(game.rounds),
        "Result ",
        str(game_wrapper.get_game_ended(game, 1)),
    )


if __name__ == "__main__":
    hparams = MCTSHparams()
    game_wrapper = BlokusGameWrapper(hparams)

    # print(game_wrapper.all_possible_indexes_to_moves[1])

    human_play(game_wrapper)
