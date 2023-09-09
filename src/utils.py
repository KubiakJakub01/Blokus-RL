"""Utility functions for the project."""
import random
import logging

import gymnasium as gym
import torch
import numpy as np
import coloredlogs

from src.hparams import HParams

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(
    coloredlogs.ColoredFormatter("%(asctime)s %(levelname)s %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def LOG_INFO(*args, **kwargs):
    """Log an info message."""
    logger.info(*args, **kwargs)


def LOG_WARNING(*args, **kwargs):
    """Log a warning message."""
    logger.warning(*args, **kwargs)


def LOG_ERROR(*args, **kwargs):
    """Log an error message."""
    logger.error(*args, **kwargs)


def seed(seed_number: int):
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True


def make_env(idx, hparams: HParams):
    def _make_env():
        env = gym.make(hparams.gym_env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if hparams.capture_video:
            if idx == 0:
                env = gym.make(hparams.gym_env, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(
                    env, f"{hparams.log_dir}/{hparams.run_name}"
                )
        env.action_space.seed(hparams.seed)
        env.observation_space.seed(hparams.seed)
        return env

    return _make_env
