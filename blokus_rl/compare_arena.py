"""Module containing the Arena class for playing games between two agents."""
import argparse
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from .blokus import BlokusGameWrapper, BlokusNNetWrapper
from .hparams import MCTSHparams, load_hparams
from .mcts import MCTS, Arena
from .utils import LOG_INFO


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


def init_player(player: str, game: BlokusGameWrapper, hparams: MCTSHparams, device: str):
    if player == "random":
        print("Using random player")
        return game.sample_move
    if Path(player).exists():
        print(f"Using player from: {player}")
        nnet = BlokusNNetWrapper(game, hparams, device)
        nnet.load_checkpoint(player)
        mcts = MCTS(game, nnet, hparams)
        return lambda x: int(np.argmax(mcts.get_action_prob(x, temp=0)))
    raise ValueError(f"Unknown player: {player}")


def log_video(hparams: MCTSHparams, items: list[dict[str, Any]], step: int):
    """Log the video."""
    if not hparams.capture_video:
        return
    LOG_INFO("Logging video")
    video_dir = hparams.video_dir / f"step_{step}"
    video_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        frames = item["frames"]
        player = item["player"]
        # Save the video
        video_fp = video_dir / f"player_{player}.mp4"
        imageio.mimsave(video_fp, frames, fps=1)


def main():
    # Get CLI arguments
    params = get_params()
    # Initialize game
    hparams = load_hparams(params.hparams_fp, "mcts")
    device = "cuda" if torch.cuda.is_available() and hparams.cuda else "cpu"
    game = BlokusGameWrapper(hparams)

    # Initialize players
    player1 = init_player(hparams.player_1, game, hparams, device)
    player2 = init_player(hparams.player_2, game, hparams, device)

    # Initialize arena
    arena = Arena(player1, player2, game, capture_video=hparams.capture_video)

    # Play games
    player1_wins, player2_wins, draws, items = arena.play_games(
        hparams.arena_compare, verbose=hparams.verbose
    )

    # Print results
    LOG_INFO("Player 1 wins: %d", player1_wins)
    LOG_INFO("Player 2 wins: %d", player2_wins)
    LOG_INFO("Draws: %d", draws)

    # Log video
    log_video(hparams, items, 0)


if __name__ == "__main__":
    main()
