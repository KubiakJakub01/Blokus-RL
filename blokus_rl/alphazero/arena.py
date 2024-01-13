"""Module with the Arena class for playing games between players."""
from itertools import permutations

import numpy as np
from tqdm import tqdm

from ..utils import log_debug, log_info


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
        verbose: Whether to LOG_DEBUG information about the game.
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
        else [np.arange(len(players))]  # type: ignore
    )

    # Initialize scoreboard
    items = []
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
        items.append({"scores": current_scores, "frames": frames})

        pbar.set_postfix(scores=scores)
        pbar.update()
    pbar.close()

    if verbose:
        log_debug("Final scores: %s", str(scores))

    return scores, items


def play_single_match(game, players, order, verbose, capture_video):
    frames = []
    s, current_player = game.get_init_board()

    if capture_video:
        frames.append(game.render(s))

    current_scores = None
    while current_scores is None:
        p = order[current_player]
        if verbose:
            log_debug("Current player: %s", str(current_player))
            log_debug("Current state:")
            log_debug(game.display(s))
        print(f"Player {p} {type(p)} thinking...",)
        s, current_player = players[p].update_state(s, current_player)

        if capture_video:
            frames.append(game.render(s))
        current_scores = game.get_game_ended(s)

    return current_scores, frames
