"""Entry point for training the model."""
import argparse
from pathlib import Path

from .alphazero.trainer import AlphaZeroTrainer
from .hparams import load_hparams
from .ppo import PPOTrainer
from .utils import log_info, seed, set_environ


def get_params():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--hparams_fp",
        type=Path,
        default=None,
        help="Path to a YAML file containing hyperparameters.",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["ppo", "alphazero"],
        default="ppo",
        help="Algorithm to use for training.",
    )
    return parser.parse_args()


def train_agent(algo: str, hparams):
    """Train a model."""
    if algo == "ppo":
        trainer = PPOTrainer(hparams)  # type: ignore
    elif algo == "alphazero":
        trainer = AlphaZeroTrainer(hparams)  # type: ignore
    trainer.train()


def main():
    # Parse command line arguments
    args = get_params()

    # Load hyperparameters and trainer
    hparams = load_hparams(args.hparams_fp, args.algorithm)

    log_info("Training model with %s algorithm", args.algorithm)
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

    log_info(hparams)

    # Set environment variables
    set_environ(hparams)
    seed(hparams.seed)

    train_agent(args.algorithm, hparams)


if __name__ == "__main__":
    main()
