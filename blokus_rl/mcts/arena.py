import logging

from tqdm import tqdm

from ..blokus.blokus_wrapper import BlokusGameWrapper

log = logging.getLogger(__name__)


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game: BlokusGameWrapper, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player1, self.player2]
        board, player = self.game.get_init_board()
        it = 0
        while self.game.get_game_ended(board, player) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(player))
                self.display(board)
            action = players[player - 1](self.game.get_canonical_form(board, player))

            valids = self.game.get_valid_moves(
                self.game.get_canonical_form(board, player)
            )

            if valids[action] == 0:
                log.error(f"Action {action} is not valid!")
                log.debug(f"valids = {valids}")
                assert valids[action] > 0
            board, player = self.game.get_next_state(board, player, action)
        if verbose:
            assert self.display
            print(
                "Game over: Turn ",
                str(it),
                "Result ",
                str(self.game.get_game_ended(board, 1)),
            )
            self.display(board)
        return self.game.get_game_ended(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        with tqdm(range(num), desc="Arena.playGames (1)") as pbar:
            for _ in range(num):
                gameResult = self.play_game(verbose=verbose)
                if gameResult == 1:
                    oneWon += 1
                elif gameResult == -1:
                    twoWon += 1
                else:
                    draws += 1
                pbar.update()
                pbar.set_postfix(oneWon=oneWon, twoWon=twoWon, draws=draws)

        self.player1, self.player2 = self.player2, self.player1

        with tqdm(range(num), desc="Arena.playGames (2)") as pbar:
            for _ in range(num):
                gameResult = self.play_game(verbose=verbose)
                if gameResult == -1:
                    oneWon += 1
                elif gameResult == 1:
                    twoWon += 1
                else:
                    draws += 1
                pbar.update()
                pbar.set_postfix(oneWon=oneWon, twoWon=twoWon, draws=draws)

        return oneWon, twoWon, draws
