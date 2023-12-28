"""Implementation of the Blokus game."""
import json
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from colosseumrl.envs.blokus import (
    GAME_PIECE_VALUES,
    PLAYER_TO_COLOR,
    BlokusEnvironment,
    Board,
    action_to_string,
)
from tqdm import tqdm

from ..hparams import AlphaZeroHparams
from ..utils import log_info, log_warning


class ColosseumBlokusGameWrapper:
    """Blokus game wrapper for the MCTS algorithm.

    This class is used to wrap the Blokus game in order to be used by the MCTS algorithm.
    """

    def __init__(self, hparams: AlphaZeroHparams):
        """Initializes the Blokus game wrapper.

        Args:
            hparams: Hyperparameters for the MCTS algorithm
        """
        self.hparams = hparams
        self.board_size = self.hparams.board_size
        self.number_of_players = self.hparams.number_of_players
        self.hparams.states_dir.mkdir(parents=True, exist_ok=True)
        self._move_action_dict: dict = {}
        self.action_move_dict: dict = {}
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
        return len(self.action_move_dict)

    def get_observation_size(self) -> list[int]:
        """
        Returns:
            number of all possible actions
        """
        return [self.number_of_players * 2, self.board_size, self.board_size]

    def get_number_of_players(self) -> int:
        """
        Returns:
            number of players
        """
        return self.number_of_players

    def get_init_board(self):
        """
        Returns:
            state: tuple of (board, round_count, players)
            player: the player who plays next
        """
        current_state, current_players = self.env.new_state()
        return current_state, current_players[0]

    def get_next_state(  # pylint: disable=unused-argument
        self, current_state: Any, current_player: int, action_id: int
    ) -> tuple[Any, int]:
        """
        Args:
            current_state: tuple of (board, round_count, players)
            player: current player index
            action: action taken by current player

        Returns:
            board after applying action
            player who plays in the next turn (should be -player)
        """
        action = self.action_move_dict[action_id]
        next_state, next_players, *_ = self.env.next_state(
            state=current_state, players=[current_player], actions=[action]
        )
        return next_state, next_players[0]

    def get_valid_moves(self, current_state: Any, current_player: int):
        """
        Args:
            current_state: tuple of (board, round_count, players)
            player: current player index

        Returns:
            mask: a binary vector of length self.get_action_size(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        if current_player == -1:
            # If the current player is -1 then get the valid actions for the next player
            current_player = current_state[0].player_color
        valid_actions = self.env.valid_actions(
            state=current_state, player=current_player
        )
        mask = np.zeros(self.get_action_size())
        if valid_actions[0] == "":
            # If there are no valid actions then return a mask of all 0s
            return mask
        for action in valid_actions:
            action_id = self._move_action_dict[action]
            mask[action_id] = 1
        return mask

    def get_observation(self, state: Any, player: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert the state to an observation.

        Args:
            state: tuple of (board, round_count, players)
            player: The player to convert the observation for

        Returns:
            observation: The observation for the player
        """
        board = state[0]
        mask = self.get_valid_moves(state, player)
        return board.canonical_board, mask

    def get_game_ended(self, state):
        """
        Args:
            current_state: tuple of (board, round_count, players)

        Returns:
            winner: 1 if player 1 won, -1 if player -1 won, 0 if draw, None if
                    game is not over
        """
        winners = self.env.get_winners(state)
        # Make one hot vector of winners
        if winners:
            # If there is a winner make a one array of -1
            one_hot_winners = np.ones(self.number_of_players) * -1
            if len(winners) == 1:
                # If there is only one winner
                one_hot_winners[winners[0]] = 3
            else:
                # If there is a draw
                for winner in winners:
                    one_hot_winners[winner] = 1
            return one_hot_winners
        return None

    def get_scores(self, winners):
        """Get the scores for the game.

        Args:
            winners: The winners of the game

        Returns:
            scores: The scores for the game
        """
        # If there is a winner make a one array of -1
        scores = np.ones(self.number_of_players) * -1
        if len(winners) == 1:
            # If there is only one winner
            scores[winners[0]] = 1
        else:
            # If there is a draw
            for winner in winners:
                scores[winner] = 0
        return scores

    def string_representation(self, state: Any) -> int:
        """
        Args:
            state: tuple of (board, round_count, players)

        Returns:
            board_string: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        board = state[0]
        return hash(board.board_contents.tobytes())

    def display(self, state: Any) -> None:
        """
        Display the current board.

        Args:
            state: tuple of (board, round_count, players)

        Returns:
            None
        """
        board = state[0]
        print(board.board_contents)

    def get_sample_move(self, state: Any):
        """
        Get a random move.

        Args:
            state: tuple of (board, round_count, players)

        Returns:
            move: a random move
        """
        players = state[-1]
        valid_actions = self.env.valid_actions(state, players[0])
        return np.random.choice(valid_actions)

    def render(self, state: Any):
        """
        Render the current board.

        Args:
            state: tuple of (board, round_count, players)
        """
        board = state[0]
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.board_size)
        ax.set_ylim(0, self.board_size)
        colors = {0: "lightgrey", 1: "red", 2: "blue", 3: "yellow", 4: "green"}

        for y in range(self.board_size):
            for x in range(self.board_size):
                polygon = plt.Polygon(
                    [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]]
                )
                polygon.set_facecolor(colors[board.board_contents[y][x].item()])
                ax.add_patch(polygon)

        plt.yticks(np.arange(0, self.board_size, 1))
        plt.xticks(np.arange(0, self.board_size, 1))
        plt.grid()

        # Render the image and convert it to a NumPy array
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)  # Close the figure to free up resources

        return image_data

    def _set_all_possible_moves(self):
        """Set all possible moves."""

        if self.states_fp.exists():
            log_info("Loading all possible states from %s", str(self.states_fp))
            with open(self.states_fp, "r", encoding="utf-8") as json_file:
                self._move_action_dict = json.load(json_file)
            self.action_move_dict = {v: k for k, v in self._move_action_dict.items()}

        else:
            log_warning("Building all possible states, this may take some time")
            board = Board(track_canonical=False)
            actions_dict = board.get_all_valid_moves(
                round_count=0,
                player_color=PLAYER_TO_COLOR[1],
                player_pieces=GAME_PIECE_VALUES,
                define_states=True,
            )

            hash_action_dict = defaultdict(list)
            move_action_dict = {}
            for piece_type, index_orientation_dict in tqdm(
                actions_dict.items(), desc="Setting all possible moves..."
            ):
                for index, orientation_list in index_orientation_dict.items():
                    for orientation in orientation_list:
                        board.reset_board()
                        board.update_board(1, piece_type, index, orientation, 0, True)
                        board_hash = hash(board.board_contents.tobytes())
                        s = action_to_string(
                            piece_type=piece_type, index=index, orientation=orientation
                        )
                        hash_action_dict[board_hash].append(s)
            for i, kv in enumerate(hash_action_dict.items()):
                board_hash, action_list = kv
                move_action_dict.update({action: i for action in action_list})

            self._move_action_dict = move_action_dict
            self.action_move_dict = {v: k for k, v in self._move_action_dict.items()}
            # Save the action dictionary to json file
            with open(self.states_fp, "w", encoding="utf-8") as json_file:
                json.dump(move_action_dict, json_file)
            log_info("%s has been saved", str(self.states_fp))
        log_info("Number of all possible states: %d", len(self.action_move_dict))
