"""Entry point for training the model."""
import argparse
from pathlib import Path


from .hparams import load_hparams, HParams
from .utils import LOG_INFO, seed, set_environ
from .ppo import Trainer
from .stablebaseline import SBTrainer


def get_params():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--hparams_fp",
        type=Path,
        required=False,
        help="Path to a YAML file containing hyperparameters.",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        choices=["ppo", "sb_ppo"],
        help="Which trainer to use.",
    )
    return parser.parse_args()


def train(hparams: HParams, trainer: str = "ppo"):
    """Train a model."""
    if trainer == "ppo":
        trainer = Trainer(hparams)
    elif trainer == "sb_ppo":
        trainer = SBTrainer(hparams)
    trainer.train()


if __name__ == "__main__":
    # Parse command line arguments
    args = get_params()

    # Load hyperparameters
    hparams = load_hparams(args.hparams_fp)

    LOG_INFO(f"Run name: {hparams.run_name}")
    # Set up wandb
    if hparams.wanda:
        import wandb

        wandb.init(
            project=hparams.wandb_project_name,
            sync_tensorboard=True,
            config=hparams,
            name=hparams.run_name,
            monitor_gym=True,
            save_code=True,
        )

    LOG_INFO(hparams)

    # Set environment variables
    set_environ(hparams)
    seed(hparams.seed)

    train(hparams, args.trainer)
