"""Module containing the Arena class for playing games between agents."""
import argparse
from pathlib import Path
from typing import Any

import imageio
import torch

from .alphazero import play_match
from .colossumrl import ColosseumBlokusGameWrapper
from .hparams import AlphaZeroHparams, load_hparams
from .neural_network import BlokusNNetWrapper
from .players import HumanPlayer, MCTSPlayer, RandomPlayer, Player
from .utils import log_info, set_environ


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--players",
        nargs="+",
        type=str,
        help="List of players to compare. \
            Can be a path to a checkpoint, \
            'mcts', 'random', or 'human'.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print additional information about the game.",
    )
    parser.add_argument(
        "--permute",
        action="store_true",
        help="Whether to permute the order of players.",
    )
    parser.add_argument(
        "--capture-video",
        action="store_true",
        help="Whether to capture a video of the game.",
    )
    parser.add_argument(
        "-n",
        "--num-games",
        type=int,
        default=1,
        help="The number of games to play.",
    )
    args = parser.parse_args()

    def _valid_args(args):
        assert len(args.players) == 4, "Must specify exactly 4 players"
        for player in args.players:
            assert (
                Path(player).exists()
                or player in ["mcts", "random", "human"]
            ), f"Invalid player: {player}"

    _valid_args(args)
    return args


def init_player(player: str, game: ColosseumBlokusGameWrapper) -> Player:
    """Initialize a player.

    Args:
        player: The player to initialize.
        game: The game to play.

    Returns:
        The initialized player.
    """
    if Path(player).exists():
        # Initialize game
        hparams = load_hparams(player, "alphazero")
        device = "cuda" if torch.cuda.is_available() and hparams.cuda else "cpu"
        nnet = BlokusNNetWrapper(game, hparams, device, "resnet")
        nnet.load_checkpoint(hparams.load_checkpoint_step)
        return MCTSPlayer(
            game=game,
            nn=nnet,
            simulations=hparams.num_mcts_sims,
        )

    if player == "mcts":
        hparams = load_hparams(algorithm="alphazero")
        device = "cuda" if torch.cuda.is_available() and hparams.cuda else "cpu"
        nnet = BlokusNNetWrapper(game, hparams, device, "dumbnet")
        return MCTSPlayer(
            game=game,
            nn=nnet,
            simulations=hparams.num_mcts_sims,
        )

    if player == "random":
        return RandomPlayer(game)

    if player == "human":
        return HumanPlayer(game)

    raise ValueError(f"Unknown player: {player}")


def log_video(hparams: AlphaZeroHparams, items: list[dict[str, Any]], step: int) -> None:
    """Log the video.

    Args:
        hparams: The hyperparameters.
        items: The items to log.
        step: The step to log.

    Returns:
        None
    """
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
    hparams = load_hparams(algorithm="alphazero")
    set_environ(hparams)
    game = ColosseumBlokusGameWrapper(hparams)

    # Initialize players
    players = [
        init_player(player, game) for player in params.players
    ]

    # Play games
    scores, items = play_match(
        game,
        players,
        games_num=params.num_games,
        permute=params.permute,
        capture_video=params.capture_video,
    )

    # Print results
    log_info("Arena compare %s: %s", str(hparams.arena_players), str(scores))

    # Log video
    log_video(hparams, items, 0)


if __name__ == "__main__":
    main()
