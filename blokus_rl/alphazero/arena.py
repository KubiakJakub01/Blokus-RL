"""Module with the Arena class for playing games between players."""
from itertools import permutations

import imageio
import numpy as np
from tqdm import tqdm

from ..utils import LOG_INFO, LOG_DEBUG


def play_match(
    game,
    players: list,
    games_num: int,
    verbose=False,
    permute=False,
    capture_video=False,
):
    """Play a match between players.

    Args:
        game: The game to play.
        players: A list of players.
        games_num: The number of games to play.
        verbose: Whether to print information about the game.
        permute: Whether to permute the order of players.
        capture_video: Whether to capture a video of the game.

    Returns:
        The scores for each player.
        Frames of the games if capture_video=True.
    """
    # You can use permutations to break the dependence on player order in measuring strength.
    matches = (
        list(permutations(np.arange(len(players))))
        if permute
        else [np.arange(len(players))]
    )

    # Initialize scoreboard
    frames_list = []
    scores = np.zeros(game.get_number_of_players())

    # Run the matches (there will be multiple if permute=True)
    # for order in tqdm(matches, desc="Playing matches"):
    pbar = tqdm(total=games_num, desc="Playing matches")
    for i in range(games_num):
        order = matches[i % len(matches)]
        for p in players:
            p.reset()  # Clear player trees to make the next match fair
        current_scores, frames = play_single_match(
            game, players, order, verbose, capture_video
        )
        scores[list(order)] += current_scores
        frames_list.append(frames)

        pbar.set_postfix(scores=scores)
        pbar.update()
    pbar.close()

    if verbose:
        LOG_DEBUG("Final scores: %s", str(scores))

    return scores, frames_list


def play_single_match(game, players, order, verbose, capture_video):
    frames = []
    s, current_player = game.get_init_board()

    if capture_video:
        frames.append(game.render(s))

    current_scores = None
    while current_scores is None:
        p = order[current_player]
        if verbose:
            print("Player #{}'s turn.".format(p))
            print("Current player: {}".format(current_player))
            print("Current state:")
            print(game.display(s))
        s, current_player = players[p].update_state(s, current_player)

        if capture_video:
            frames.append(game.render(s))
        current_scores = game.get_game_ended(s)

    return current_scores, frames


if __name__ == "__main__":
    from ..colossumrl import BlokusNNet, ColosseumBlokusGameWrapper
    from ..hparams import MCTSHparams
    from ..models import DumbNet
    from ..neural_network import BlokusNNetWrapper
    from ..players import MCTSPlayer

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
    scores, frames = play_match(
        game, [agent] + uninformeds, permute=False, verbose=False, capture_video=True
    )
    print(f"Opponent strength: {opponent_sims}\nScores: {scores}")

    # Save the video
    video_fp = hparams.video_dir / f"colloseumrl_debug.mp4"
    print("Saving video to %s", str(video_fp))
    print(f"Frames: {len(frames)}")
    imageio.mimsave(video_fp, frames, fps=1)
