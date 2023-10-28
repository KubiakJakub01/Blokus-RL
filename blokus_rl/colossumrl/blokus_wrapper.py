"""Implementation of the Blokus game."""
import itertools
import json
import multiprocessing as mp
from functools import partial
from typing import Any

import cython
import numpy as np
from colosseumrl.envs.blokus import (GAME_PIECE_VALUES,
                                    PLAYER_TO_COLOR,
                                    BlokusEnvironment,
                                    Board, action_to_string)
from tqdm import tqdm

from ..hparams import MCTSHparams
from ..utils import LOG_INFO, LOG_WARNING


class ColosseumBlokusGameWrapper:
    """Blokus game wrapper for the MCTS algorithm.

    This class is used to wrap the Blokus game in order to be used by the MCTS algorithm.
    """


    def __init__(self, hparams: MCTSHparams):
        """Initializes the Blokus game wrapper.

        Args:
            hparams: Hyperparameters for the MCTS algorithm
        """
        self.hparams = hparams
        self.board_size = self.hparams.board_size
        self.number_of_players = self.hparams.number_of_players
        self.hparams.states_dir.mkdir(parents=True, exist_ok=True)
        self.all_possible_indexes_to_moves: list = []
        self.starter_won = 0
        self.last_won = 0
        self.games_played = 0
        self.env = BlokusEnvironment()
        self._set_all_possible_moves()

    @property
    def states_fp(self):
        return (
            self.hparams.states_dir
            / f"colosseum_{self.board_size}_players_{self.number_of_players}.json"
        )
    
    def get_init_board(self):
        """
        Returns:
            blokus_game: a BlokusGame object
            player: the player who plays next
        """
        current_state, current_players = self.env.new_state()
        return current_state, current_players

    def get_board_size(self) -> tuple[int, int]:
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (20, 20)

    def get_action_size(self) -> int:
        """
        Returns:
            number of all possible actions
        """
        return len(self.all_possible_indexes_to_moves)

    def get_next_state(  # pylint: disable=unused-argument
        self, current_state: Any, current_players: int, action_id: int
    ) -> tuple[Any, int]:
        """
        Args:
            blokus_game: blokus game object
            player: current player index
            action: action taken by current player

        Returns:
            board after applying action
            player who plays in the next turn (should be -player)
        """
        action = self.all_possible_indexes_to_moves[action_id]
        next_state, next_players, rewards, terminal, winners = self.env.next_state(
            state=current_state,players=current_players,actions=[action])
        return next_state, next_players

    def get_valid_moves(self, current_state: Any, current_players) -> list[int]:
        """
        Args:
            blokus_game: blokus game object

        Returns:
            mask: a binary vector of length self.get_action_size(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valid_actions = self.env.valid_actions(
            state=current_state, player=current_players
        )
        mask = np.zeros(self.get_action_size())
        for action in valid_actions:
            action_id = self.all_possible_indexes_to_moves.index(action)
            mask[action_id] = 1
        return mask

    def get_game_ended(
        self, state, players, verbose: bool = False
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
        winners = self.env.get_winners(state)
        print(f'{winners=}')
        # Make one hot vector of winners
        if winners:
            # If there is a winner make a one array of -1
            one_hot_winners = np.ones(self.number_of_players) * -1
            if len(winners) == 1:
                # If there is only one winner
                one_hot_winners[winners[0]] = 1
            else:
                # If there is a draw
                for winner in winners:
                    one_hot_winners[winner] = 0
            return one_hot_winners
        return None

    def get_canonical_form(self, state: Any, player: int) -> Any:
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
        return state

    def get_symmetries(
        self, state: Any, pi: list[int]
    ) -> list[tuple[Any, list[int]]]:
        """
        Args:
            blokus_game: blokus game object
            pi: policy vector of size self.get_action_size()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        board, *_ = state
        return [(board.canonical_board, pi)]


    def string_representation(self, state: Any) -> str:
        """
        Args:
            blokus_game: blokus game object

        Returns:
            board_string: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        board, *_ = state
        return hash(board.board_contents.tobytes())

    def sample_move(self, blokus_game: Any):
        """
        Get a random move.

        Args:
            blokus_game: blokus game object

        Returns:
            move: a random move
        """
        pass

    def display(self, state: Any, mode: str = "tensor") -> None:
        """
        Display the current board.

        Args:
            blokus_game: blokus game object

        Returns:
            None
        """
        board, *_ = state
        print(board.board_contents)

    def render(self, blokus_game: Any):
        """
        Render the current board.

        Args:
            blokus_game: blokus game object
        """
        pass
    
    def _set_all_possible_moves(self):
        """Set all possible moves."""

        if self.states_fp.exists():
            LOG_INFO("Loading all possible states from %s", str(self.states_fp))
            with open(self.states_fp, "r", encoding="utf-8") as json_file:
                self.all_possible_indexes_to_moves = json.load(json_file)

        else:
            LOG_WARNING("Building all possible states, this may take some time")
            board = Board(track_canonical=False)
            actions_dict = board.get_all_valid_moves(
                round_count=0,
                player_color=PLAYER_TO_COLOR[1],
                player_pieces=GAME_PIECE_VALUES,
                define_states=True
            )

            board_list = []
            action_list = []
            for piece_type, index_orientation_dict in tqdm(actions_dict.items(), desc="Setting all possible moves..."):
                for index, orientation_list in index_orientation_dict.items():
                    for orientation in orientation_list:
                        board.reset_board()
                        board.update_board(1, piece_type, index, orientation, 0, True)
                        board_contents = hash(board.board_contents.tobytes())
                        if board_contents not in board_list:
                                s = action_to_string(piece_type=piece_type, index=index, orientation=orientation)
                                board_list.append(board_contents)
                                action_list.append(s)

            self.all_possible_indexes_to_moves = action_list
            # Save the action dictionary to json file
            with open(self.states_fp, "w", encoding="utf-8") as json_file:
                json.dump(action_list, json_file)
            LOG_INFO("%s has been saved", str(self.states_fp))
        LOG_INFO("Number of all possible states: %d", len(self.all_possible_indexes_to_moves))
