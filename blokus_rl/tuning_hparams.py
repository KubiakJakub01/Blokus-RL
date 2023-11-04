"""Script to tune hyperparameters for the Blokus environment."""

import argparse
from functools import partial
from pathlib import Path
from statistics import mean

import optuna
import torch
from torch.utils.data import DataLoader

from .alphazero.dataset import MCTSDataset
from .colossumrl import ColosseumBlokusGameWrapper
from .hparams import MCTSHparams
from .neural_network import BlokusNNetWrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/train"),
    )
    parser.add_argument(
        "--val_data_dir",
        type=Path,
        default=Path("data/valid"),
    )
    return parser.parse_args()


def train_and_evaluate_model(hparams: MCTSHparams, device: str = "cuda"):
    """Train and evaluate model."""

    def _train_epoch(nnet, train_dl: DataLoader):
        """Train for one epoch."""
        losses = []
        for batch in train_dl:
            loss = nnet.train_step(batch)
            losses.append(loss)
        return mean(losses)

    game = ColosseumBlokusGameWrapper(hparams)
    nnet = BlokusNNetWrapper(game, hparams, device=device)
    train_dl = DataLoader(
        MCTSDataset(hparams), batch_size=hparams.batch_size, shuffle=True
    )
    val_dl = DataLoader(
        MCTSDataset(hparams, is_train=False), batch_size=hparams.batch_size
    )

    train_losses = []
    for _ in range(hparams.epochs):
        train_loss = _train_epoch(nnet, train_dl)
        train_losses.append(train_loss)

    with torch.inference_mode():
        eval_losses = []
        for batch in val_dl:
            loss = nnet.eval_fn(batch)
            eval_losses.append(loss.item())

    return mean(train_losses), mean(eval_losses)


def objective(
    trial,
    data_dir: Path = Path("data/train"),
    val_data_dir: Path = Path("data/valid"),
    device: str = "cuda",
):
    # Define the hyperparameters to be tuned
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 5)
    epochs = trial.suggest_int("epochs", 10, 100)

    # Prepare the hparams object
    hparams = MCTSHparams(
        data_dir=data_dir,
        val_data_dir=val_data_dir,
        epochs=epochs,
        num_res_blocks=num_layers,
        batch_size=batch_size,
        lr=lr,
    )

    # Train and evaluate the model
    train_loss, val_loss = train_and_evaluate_model(hparams, device=device)

    # Return the validation accuracy as the objective to be minimized
    return val_loss


if __name__ == "__main__":
    # Parse command line arguments
    args = get_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimize hyperparameters
    study = optuna.create_study(direction="minimize")
    _objective = partial(
        objective, data_dir=args.data_dir, val_data_dir=args.val_data_dir, device=device
    )
    study.optimize(_objective, n_trials=100)
    print(study.best_trial)
