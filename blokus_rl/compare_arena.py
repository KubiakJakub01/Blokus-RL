"""Module containing the Arena class for playing games between agents."""
import argparse
from pathlib import Path
from typing import Any

import imageio
import torch

from .colossumrl import ColosseumBlokusGameWrapper
from .hparams import AlphaZeroHparams, load_hparams
from .neural_network import BlokusNNetWrapper
from .players import MCTSPlayer
from .utils import log_info, set_environ
from .alphazero import play_match


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hparams_fp",
        type=Path,
        default="blokus_rl/configs/mcts.yaml",
        help="Path to hyperparameter file",
    )
    args = parser.parse_args()
    return args


def init_player(
    player: str, game: ColosseumBlokusGameWrapper, hparams: AlphaZeroHparams, device: str
):
    """Initialize a player.

    Args:
        player: The player to initialize.
        game: The game.
        hparams: The hyperparameters.
        device: The device to use.

    Returns:
        A function that takes a board as input and returns an action."""
    if Path(player).exists():
        nnet = BlokusNNetWrapper(game, hparams, device, "resnet")
        nnet.load_checkpoint(player)
        return MCTSPlayer(
            game=game,
            nn=nnet,
            simulations=hparams.num_mcts_sims,
        )

    if player == "uninformed":
        nnet = BlokusNNetWrapper(game, hparams, device, "dumbnet")
        return MCTSPlayer(
            game=game,
            nn=nnet,
            simulations=hparams.opponent_strength,
        )

    if player == "random":
        raise NotImplementedError("Random player not implemented")

    if player == "human":
        raise NotImplementedError("Human player not implemented")

    raise ValueError(f"Unknown player: {player}")


def log_video(hparams: AlphaZeroHparams, items: list[dict[str, Any]], step: int):
    """Log the video."""
    if not hparams.capture_video:
        return
    log_info("Logging video")
    video_dir = hparams.video_dir / f"eval_{step}"
    video_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(items, 1):
        # Save the video
        frames = item["frames"]
        scores = item["scores"]
        video_fp = video_dir / f"game_{i}_scores_{scores}.mp4"
        imageio.mimsave(video_fp, frames, fps=1)


def main():
    # Get CLI arguments
    params = get_params()

    # Initialize game
    hparams = load_hparams(params.hparams_fp, "mcts")
    set_environ(hparams)
    device = "cuda" if torch.cuda.is_available() and hparams.cuda else "cpu"
    game = ColosseumBlokusGameWrapper(hparams)

    # Initialize players
    players = [
        init_player(player, game, hparams, device) for player in hparams.arena_players
    ]

    # Play games
    scores, items = play_match(
        game,
        players,
        games_num=hparams.compare_arena_games,
        permute=hparams.permute,
        capture_video=hparams.capture_video,
    )

    # Print results
    log_info("Arena compare %s: %s", str(hparams.arena_players), str(scores))

    # Log video
    log_video(hparams, items, 0)


if __name__ == "__main__":
    main()
