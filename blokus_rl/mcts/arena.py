"""Module containing the Arena class for playing games between two agents."""

from tqdm import tqdm

from ..blokus.blokus_wrapper import BlokusGameWrapper
from ..utils import LOG_INFO, LOG_WARNING


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game: BlokusGameWrapper, capture_video=False):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            capture_video: whether to capture video of the game
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.capture_video = capture_video

    def play_game(self, verbose=False, capture_video=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        frames = []
        players = [self.player1, self.player2]
        board, player = self.game.get_init_board()
        it = 0
        if capture_video:
            frames.append(self.game.render(board))
        while self.game.get_game_ended(board, player) == 0:
            it += 1
            action = players[player - 1](self.game.get_canonical_form(board, player))

            valids = self.game.get_valid_moves(
                self.game.get_canonical_form(board, player)
            )

            if valids[action] == 0:
                LOG_WARNING(f"Action {action} is not valid!")
                LOG_INFO(f"valids = {valids}")
                assert valids[action] > 0
            board, player = self.game.get_next_state(board, player, action)
            if verbose:
                LOG_INFO("Turn ", str(it), "Player ", str(player))
                self.game.display(board)
            if capture_video:
                frames.append(self.game.render(board))
        if verbose:
            LOG_INFO(
                "Game over: Turn %s Result %s",
                str(it),
                str(self.game.get_game_ended(board, 1, verbose=verbose)),
            )
            self.game.display(board)
        return self.game.get_game_ended(board, 1), frames

    def play_games(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = num // 2
        one_won = 0
        two_won = 0
        draws = 0
        items = []

        with tqdm(range(num), desc="Arena.playGames (1)") as pbar:
            for _ in range(num):
                game_result, _ = self.play_game(verbose=verbose)
                if game_result == 1:
                    one_won += 1
                elif game_result == -1:
                    two_won += 1
                else:
                    draws += 1
                pbar.update()
                pbar.set_postfix(oneWon=one_won, twoWon=two_won, draws=draws)

        if self.capture_video:
            game_result, frames = self.play_game(verbose=verbose, capture_video=True)
            items.append({"frames": frames, "result": game_result, "player": 1})

        self.player1, self.player2 = self.player2, self.player1

        with tqdm(range(num), desc="Arena.playGames (2)") as pbar:
            for _ in range(num):
                game_result, _ = self.play_game(verbose=verbose)
                if game_result == -1:
                    one_won += 1
                elif game_result == 1:
                    two_won += 1
                else:
                    draws += 1
                pbar.update()
                pbar.set_postfix(oneWon=one_won, twoWon=two_won, draws=draws)

        if self.capture_video:
            game_result, frames = self.play_game(verbose=verbose, capture_video=True)
            items.append({"frames": frames, "result": game_result, "player": 2})

        return one_won, two_won, draws, items
