"""Utility functions for the project."""
import logging
import random

import coloredlogs
import gymnasium as gym
import numpy as np
import torch

from .hparams import HParams

# Set up logging
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("debug.log", mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
file_handler.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(
    coloredlogs.ColoredFormatter("%(asctime)s %(levelname)s %(message)s")
)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(file_handler)


def LOG_DEBUG(*args, **kwargs):
    """Log an debug message."""
    logger.debug(*args, **kwargs)


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


def set_environ(hparams: HParams):
    """Set environment variables."""
    logger.setLevel(hparams.log_level)
    torch.autograd.set_detect_anomaly(hparams.detect_anomaly)


def make_envs(idx, hparams: HParams):
    def _make_env():
        env = gym.make(hparams.gym_env, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if hparams.capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, f"{hparams.log_dir}/{hparams.run_name}"
                )
        env.action_space.seed(hparams.seed)
        env.observation_space.seed(hparams.seed)
        return env

    return _make_env


class AverageMeter:
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{self.avg:.2e}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
