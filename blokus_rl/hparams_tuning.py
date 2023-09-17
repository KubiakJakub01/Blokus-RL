"""Script to tune hyperparameters with Optuna."""
import argparse
import logging
from pathlib import Path

import optuna
import yaml
from optuna.trial import Trial

from .hparams import HParams
from .utils import LOG_INFO
from .ppo import Trainer


def objective(trial: Trial) -> float:
    """Objective function for Optuna to optimize.

    Args:
        trial (Trial): Optuna trial object.

    Returns:
        float: Negative of the mean episode reward.
    """
    # Sample hyperparameters
    hparams_dict = {
        "num_envs": trial.suggest_int("num_envs", 1, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "total_timesteps": trial.suggest_int("total_timesteps", 1000, 1e6, log=True),
        "num_steps": trial.suggest_int("num_steps", 1, 128),
        "update_epochs": trial.suggest_int("update_epochs", 1, 10),
        "d_model": trial.suggest_int("d_model", 16, 512, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.999),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.5),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 1.0),
        "seed": trial.suggest_int("seed", 0, 1000),
    }

    # Create hparams object
    hparams_obj = HParams(**hparams_dict)

    # Create trainer
    trainer = Trainer(hparams_obj)

    # Train
    trainer.train()

    # Return the mean episode reward
    return trainer.mean_episode_reward
