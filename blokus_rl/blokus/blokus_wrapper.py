"""Implementation of the Blokus game."""
import itertools
import json
import multiprocessing as mp
import os
from functools import partial

import cython

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
    
    This class is used to wrap the Blokus game in order to be used by the MCTS algorithm."""

    rewards = {"won": 1, "tie-won": 0.1, "default": 0, "invalid": -100, "lost": -1}

    def __init__(self, board_size: int, number_of_players: int, states_fp=None):
        """Initializes the Blokus game wrapper.
        
        Args:
            board_size: An integer representing the size of the board.
            number_of_players: An integer representing the number of players.
            states_fp: A string representing the path to the file where the states are saved."""
        if not cython.compiled:
            print("You should run 'python setup.py build_ext --inplace' to get a 3x speedup")
        self.STATES_FOLDER = "states"
        self.all_possible_indexes_to_moves = None
        self.starter_won = 0
        self.last_won = 0
        self.games_played = 0
        self.BOARD_SIZE = board_size
        self.NUMBER_OF_PLAYERS = number_of_players
        self.states_fp = states_fp
        self.all_shapes = get_all_shapes()
        self._set_all_possible_moves()

    def getInitBoard(self) -> tuple[BlokusGame, int]:
        """
        Returns:
            blokus_game: a BlokusGame object
            player: the player who plays next
        """
        board = Board(self.BOARD_SIZE)
        blokus_game = BlokusGame(board, self.all_shapes)
        for i in range(1, self.NUMBER_OF_PLAYERS + 1):
            blokus_game.add_player(
                AiPlayer(
                    i, f"Player {i}", self.all_possible_indexes_to_moves, blokus_game
                )
            )
        return blokus_game, blokus_game.next_player().index

    def getBoardSize(self) -> tuple[int, int]:
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.BOARD_SIZE, self.BOARD_SIZE)

    def getActionSize(self) -> int:
        """
        Returns:
            number of all possible actions
        """
        return len(self.all_possible_indexes_to_moves)

    def getNextState(self, blokus_game: BlokusGame, player: int, action: int):
        """
        Input:
            board: current board
            player: current player index
            action: action taken by current player

        Returns:
            board after applying action
            player who plays in the next turn (should be -player)
        """
        if action == -1:
            print(
                f"Player {blokus_game.next_player().name} invalid move but have {blokus_game.next_player().remains_move}"
            )
            blokus_game.move_to_next_player()
            return blokus_game, blokus_game.next_player().index
        move_index = self.all_possible_indexes_to_moves[action]
        blokus_game.next_player().next_move = move_index
        blokus_game.play()
        return blokus_game, blokus_game.next_player().index

    def getValidMoves(self, blokus_game: BlokusGame, player: int) -> list[int]:
        """
        Input:
            board: current board
            player: current player

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

    def getGameEnded(self, blokus_game, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        winners = blokus_game.winners()
        done = winners is not None
        if done:
            # print(f"Game over! And the winner is: {winners}")
            if player in winners:
                if len(winners) == 1:
                    reward = self.rewards["won"]
                else:
                    reward = self.rewards["tie-won"]
            else:
                reward = self.rewards["lost"]
            # print(f"Player {player} winners: {winners} reward: {reward}")
        else:
            # reward = self.rewards['default'] if self.ai.next_move is None else self.ai.next_move.size
            reward = self.rewards["default"]

        return reward

    def getCanonicalForm(self, blokus_game, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

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

    def getSymmetries(self, blokus_game, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(blokus_game.canonical_board, pi)]

    def stringRepresentation(self, blokus_game):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(blokus_game.canonical_board)

    def sample_move(self, board):
        """
        Input:
            board: current board

        Returns:
            move: a random move
        """
        return board.next_player().sample_move_idx()

    def _set_all_possible_moves(self):
        # if self.all_possible_indexes_to_moves is not None:
        #     return

        if os.path.exists(self.states_fp):
            with open(self.states_fp) as json_file:
                self.all_possible_indexes_to_moves = [
                    Shape.from_json(move) for move in json.load(json_file)
                ]
        else:
            print("Building all possible states, this may take some time")
            board = Board(self.BOARD_SIZE)
            blokus_game = BlokusGame(board, self.all_shapes, self.NUMBER_OF_PLAYERS)
            dummy = Player("", "", self.all_shapes, blokus_game)

            # self.all_possible_indexes_to_moves = possible_moves_func(dummy, self.BOARD_SIZE, self.all_shapes)
            number_of_cores_to_use = mp.cpu_count() // 2
            with mp.Pool(number_of_cores_to_use) as pool:
                self.all_possible_indexes_to_moves = pool.map(
                    partial(possible_moves_func, dummy, self.BOARD_SIZE),
                    [[p] for p in self.all_shapes],
                )
            self.all_possible_indexes_to_moves = list(
                itertools.chain.from_iterable(self.all_possible_indexes_to_moves)
            )
            data = [
                move.to_json(idx)
                for idx, move in enumerate(self.all_possible_indexes_to_moves)
            ]

            os.makedirs(self.STATES_FOLDER, exist_ok=True)
            with open(self.states_fp, "w") as json_file:
                json.dump(data, json_file)
            print(f"{self.states_fp} has been saved")
