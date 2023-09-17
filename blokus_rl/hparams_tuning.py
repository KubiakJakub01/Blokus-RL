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


def main() -> None:
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Script to tune hyperparameters with Optuna."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials to run.",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="ppo",
        help="Name of the Optuna study.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna.db",
        help="Database URL for Optuna to store results.",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        help="Pruner to use for Optuna.",
    )
    parser.add_argument(
        "--pruner_n_startup_trials",
        type=int,
        default=5,
        help="Number of trials to run before pruning begins.",
    )
    parser.add_argument(
        "--pruner_n_warmup_steps",
        type=int,
        default=0,
        help="Number of steps to run before pruning begins.",
    )
    parser.add_argument(
        "--pruner_interval_steps",
        type=int,
        default=1,
        help="Number of steps between pruning.",
    )
    parser.add_argument(
        "--pruner_n_min_trials",
        type=int,
        default=1,
        help="Minimum number of trials to run before pruning begins.",
    )
    parser.add_argument(
        "--pruner_n_max_trials",
        type=int,
        default=100,
        help="Maximum number of trials to run before pruning begins.",
    )
    parser.add_argument(
        "--pruner_n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run for pruning.",
    )
    parser.add_argument(
        "--pruner_n_points",
        type=int,
        default=100,
        help="Number of points to sample for pruning.",
    )
    parser.add_argument(
        "--pruner_n_quantiles",
        type=int,
        default=100,
        help="Number of quantiles to sample for pruning.",
    )
    parser.add_argument(
        "--pruner_bandwidth",
        type=float,
        default=1.0,
        help="Bandwidth for kernel density estimation for pruning.",
    )
    parser.add_argument(
        "--pruner_multivariate",
        type=bool,
        default=False,
        help="Whether to use multivariate kernel density estimation for pruning.",
    )
    parser.add_argument(
        "--pruner_censoring",
        type=bool,
        default=True,
        help="Whether to use censoring for pruning.",
    )
    parser.add_argument(
        "--pruner_independent_coords",
        type=bool,
        default=False,
        help="Whether to use independent coordinates for pruning.",
    )
    parser.add_argument(
        "--pruner_early_stopping",
        type=bool,
        default=True,
        help="Whether to use early stopping for pruning.",
    )
    parser.add_argument(
        "--pruner_pure",
        type=bool,
        default=False,
        help="Whether to use pure Python for pruning.",
    )
    parser.add_argument(
        "--pruner_storage",
        type=str,
        default=None,
        help="Database URL for Optuna to store pruning results.",
    )
    parser.add_argument(
        "--pruner_study_name",
        type=str,
        default=None,
        help="Name of the Optuna study for pruning.",
    )
    parser.add_argument(
        "--pruner_direction",
        type=str,
        default="minimize",
        help="Direction to optimize for pruning.",
    )

    args = parser.parse_args()

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_n_startup_trials,
            n_warmup_steps=args.pruner_n_warmup_steps,
            interval_steps=args.pruner_interval_steps,
            n_min_trials=args.pruner_n_min_trials,
            n_max_trials=args.pruner_n_max_trials,
            n_jobs=args.pruner_n_jobs,
            n_points=args.pruner_n_points,
            n_quantiles=args.pruner_n_quantiles,
            bandwidth=args.pruner_bandwidth,
            multivariate=args.pruner_multivariate,
            censoring=args.pruner_censoring,
            independent_coords=args.pruner_independent_coords,
            early_stopping=args.pruner_early_stopping,
            pure=args.pruner_pure,
            storage=args.pruner_storage,
            study_name=args.pruner_study_name,
            direction=args.pruner_direction,
        ),
    )

    # Run trials
    study.optimize(objective, n_trials=args.n_trials)

    # Print best trial
    LOG_INFO(f"Best trial: {study.best_trial.value}")

    # Print best trial parameters
    LOG_INFO("Best trial parameters:")

    for key, value in study.best_trial.params.items():
        LOG_INFO(f"    {key}: {value}")

    # Save best trial parameters to file
    best_trial_params_fp = Path("best_trial_params.yaml")

    with open(best_trial_params_fp, "w", encoding="utf-8") as f:
        yaml.dump(study.best_trial.params, f, default_flow_style=False)

    
if __name__ == "__main__":
    main()
