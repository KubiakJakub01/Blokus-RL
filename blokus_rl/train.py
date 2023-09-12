"""Entry point for training the model."""
import argparse
from pathlib import Path


from .hparams import load_hparams, HParams
from .utils import LOG_INFO, seed, set_environ
from .ppo import Trainer


def get_params():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--hparams_fp",
        type=Path,
        required=False,
        help="Path to a YAML file containing hyperparameters.",
    )
    return parser.parse_args()


def train(hparams: HParams):
    """Train a model."""
    trainer = Trainer(hparams)
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

    train(hparams)
