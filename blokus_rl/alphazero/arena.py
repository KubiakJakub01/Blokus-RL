"""Module with the Arena class for playing games between players."""
from itertools import permutations
import numpy as np
from tqdm import tqdm

from .colossumrl import ColosseumBlokusGameWrapper
from .players import Player


def play_match(
    game: ColosseumBlokusGameWrapper,
    players: list[Player],
    verbose=False,
    permute=False,
):
    # You can use permutations to break the dependence on player order in measuring strength.
    matches = (
        list(permutations(np.arange(len(players))))
        if permute
        else [np.arange(len(players))]
    )

    # Initialize scoreboard
    scores = np.zeros(game.get_number_of_players())

    # Run the matches (there will be multiple if permute=True)
    for order in tqdm(matches, desc="Playing matches"):
        for p in players:
            p.reset()  # Clear player trees to make the next match fair

        s, current_player = game.get_init_board()
        if verbose:
            game.display(s)
        terminal = False
        winners = None

        while not terminal:
            p = order[current_player]
            if verbose:
                print("Player #{}'s turn.".format(p))
            s, current_player, terminal, winners = players[p].update_state(
                s, current_player, terminal, winners
            )
            if verbose:
                game.display(s)

        scores = game.get_scores(winners)
        scores[list(order)] += scores
        if verbose:
            print(
                "Δ" + str(scores[list(order)]) + ", Current scoreboard: " + str(scores)
            )

    if verbose:
        print("Final scores:", scores)
    return scores


if __name__ == "__main__":
    from .players import MCTSPlayer
    from .hparams import MCTSHparams
    from .colossumrl import BlokusNNet
    from .models import DumbNet
    from .neural_network import BlokusNNetWrapper

    hparams = MCTSHparams()
    game = ColosseumBlokusGameWrapper(hparams)
    opponent_sims = 20
    # device = "cuda" if hparams.cuda else "cpu"
    device = "cpu"
    print(f"Using device: {device}")
    uninformeds = [
        MCTSPlayer(
            game=game,
            nn=BlokusNNetWrapper(game, hparams, DumbNet, device),
            simulations=opponent_sims,
        )
        for _ in range(3)
    ]
    agent = MCTSPlayer(
        game=game,
        nn=BlokusNNetWrapper(game, hparams, BlokusNNet, device),
        simulations=hparams.num_mcts_sims,
    )
    scores = play_match(game, [agent] + uninformeds, permute=True)
    print(f"Opponent strength: {opponent_sims}\nScores: {scores}")
