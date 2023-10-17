"""Implementation of the Blokus game."""
import itertools
import json
import multiprocessing as mp
from functools import partial
from typing import Literal

import cython
import matplotlib.pyplot as plt

from ..hparams import MCTSHparams
from ..utils import LOG_INFO, LOG_WARNING
from .game.blokus_game import BlokusGame
from .game.board import Board
from .players.player import AiPlayer, Player
from .shapes.shape import Shape
from .shapes.shapes import get_all_shapes


def possible_moves_func(dummy, board_size, pieces):
    # This needs to be there because it can't be pickled
    return dummy.possible_moves(pieces, no_restriction=True, board_size=board_size)


class BlokusGameWrapper:
    """Blokus game wrapper for the MCTS algorithm.

    This class is used to wrap the Blokus game in order to be used by the MCTS algorithm.
    """

    rewards = {"won": 1, "tie-won": 0.1, "default": 0, "invalid": -100, "lost": -1}

    def __init__(self, hparams: MCTSHparams):
        """Initializes the Blokus game wrapper.

        Args:
            hparams: Hyperparameters for the MCTS algorithm
        """
        if not cython.compiled:
            LOG_WARNING(
                "You should run 'python setup.py build_ext --inplace' to get a 3x speedup"
            )
        self.hparams = hparams
        self.board_size = self.hparams.board_size
        self.number_of_players = self.hparams.number_of_players
        self.hparams.states_dir.mkdir(parents=True, exist_ok=True)
        self.all_possible_indexes_to_moves = None
        self.starter_won = 0
        self.last_won = 0
        self.games_played = 0
        self.all_shapes = get_all_shapes()
        self._set_all_possible_moves()

    @property
    def states_fp(self):
        return (
            self.hparams.states_dir
            / f"board_{self.board_size}_players_{self.number_of_players}.json"
        )

    def get_init_board(self) -> tuple[BlokusGame, int]:
        """
        Returns:
            blokus_game: a BlokusGame object
            player: the player who plays next
        """
        board = Board(self.board_size)
        blokus_game = BlokusGame(board, self.all_shapes)
        for i in range(1, self.number_of_players + 1):
            blokus_game.add_player(
                AiPlayer(
                    i, f"Player_{i}", self.all_possible_indexes_to_moves, blokus_game
                )
            )
        return blokus_game, blokus_game.next_player().index

    def get_board_size(self) -> tuple[int, int]:
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.board_size, self.board_size)

    def get_action_size(self) -> int:
        """
        Returns:
            number of all possible actions
        """
        return len(self.all_possible_indexes_to_moves)

    def get_next_state(
        self, blokus_game: BlokusGame, player: int, action: int
    ) -> tuple[BlokusGame, int]:
        """
        Args:
            blokus_game: blokus game object
            player: current player index
            action: action taken by current player

        Returns:
            board after applying action
            player who plays in the next turn (should be -player)
        """
        if action == -1:
            LOG_WARNING(
                "Player %s invalid move but have %s moves left",
                {blokus_game.next_player().name},
                len(blokus_game.next_player().possible_move_indexes()),
            )
            blokus_game.move_to_next_player()
            return blokus_game, blokus_game.next_player().index
        move_index = self.all_possible_indexes_to_moves[action]
        blokus_game.next_player().next_move = move_index
        blokus_game.play()
        return blokus_game, blokus_game.next_player().index

    def get_valid_moves(self, blokus_game: BlokusGame) -> list[int]:
        """
        Args:
            blokus_game: blokus game object

        Returns:
            mask: a binary vector of length self.get_action_size(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        indexes = blokus_game.next_player().possible_move_indexes()
        mask = [0] * self.get_action_size()
        for index in indexes:
            mask[index] = 1
        return mask

    def get_game_ended(
        self, blokus_game: BlokusGame, player: int | None = None, verbose=False
    ):
        """
        Args:
            blokus_game: blokus game object
            player: current player index
            verbose: whether to print the winner

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        if player is None:
            player = blokus_game.next_player().index
        winners = blokus_game.winners()
        done = winners is not None
        if done:
            if player in winners:
                if len(winners) == 1:
                    reward = self.rewards["won"]
                else:
                    reward = self.rewards["tie-won"]
            else:
                reward = self.rewards["lost"]
        else:
            reward = self.rewards["default"]

        if verbose:
            LOG_INFO(f"Player {player} winners: {winners} reward: {reward}")

        return reward

    def get_canonical_form(self, blokus_game: BlokusGame, player: int) -> BlokusGame:
        """
        Args:
            blokus_game: blokus game object
            player: current player index

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        board = blokus_game.board.tensor.numpy().copy()
        index = player
        board[(board != 0) & (board != index)] = -1
        board[board == index] = 1
        blokus_game.canonical_board = board
        return blokus_game

    def get_symmetries(
        self, blokus_game: BlokusGame, pi: list[int]
    ) -> list[tuple[BlokusGame, list[int]]]:
        """
        Args:
            blokus_game: blokus game object
            pi: policy vector of size self.get_action_size()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(blokus_game.canonical_board, pi)]

    def string_representation(self, blokus_game: BlokusGame) -> str:
        """
        Args:
            blokus_game: blokus game object

        Returns:
            board_string: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(blokus_game.canonical_board)

    def sample_move(self, blokus_game: BlokusGame):
        """
        Get a random move.

        Args:
            blokus_game: blokus game object

        Returns:
            move: a random move
        """
        return blokus_game.next_player().sample_move_idx()

    def display(
        self, blokus_game: BlokusGame, mode: str = "tensor"
    ) -> None:
        """
        Display the current board.

        Args:
            blokus_game: blokus game object

        Returns:
            None
        """
        blokus_game.board.display(mode)

    def render(self, blokus_game: BlokusGame):
        """
        Render the current board.

        Args:
            blokus_game: blokus game object
        """
        return blokus_game.board.fancy_board()

    def _set_all_possible_moves(self):
        """Set all possible moves."""

        if self.states_fp.exists():
            LOG_INFO("Loading all possible states from %s", str(self.states_fp))
            with open(self.states_fp) as json_file:
                self.all_possible_indexes_to_moves = [
                    Shape.from_json(move) for move in json.load(json_file)
                ]
        else:
            LOG_WARNING("Building all possible states, this may take some time")
            board = Board(self.board_size)
            blokus_game = BlokusGame(board, self.all_shapes, self.number_of_players)
            dummy = Player(1, "", self.all_shapes, blokus_game)

            # self.all_possible_indexes_to_moves = possible_moves_func(dummy, self.BOARD_SIZE, self.all_shapes)
            number_of_cores_to_use = mp.cpu_count() // 2
            with mp.Pool(number_of_cores_to_use) as pool:
                self.all_possible_indexes_to_moves = pool.map(
                    partial(possible_moves_func, dummy, self.board_size),
                    [[p] for p in self.all_shapes],
                )
            self.all_possible_indexes_to_moves = list(
                itertools.chain.from_iterable(self.all_possible_indexes_to_moves)
            )
            data = [
                move.to_json(idx)
                for idx, move in enumerate(self.all_possible_indexes_to_moves)
            ]

            with open(self.states_fp, "w") as json_file:
                json.dump(data, json_file)
            LOG_INFO("%s has been saved", str(self.states_fp))
