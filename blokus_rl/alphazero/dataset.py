"""Module with dataset classes for MCTS"""
from pickle import Unpickler

import torch
from torch.utils.data import Dataset

from ..hparams import MCTSHparams
from ..utils import LOG_INFO


class MCTSDataset(Dataset):
    """Dataset for MCTS"""

    def __init__(self, hparams: MCTSHparams):
        super().__init__()

        # Get hyperparameters
        self.hparams = hparams

        # Load data from files
        self.data = self.load_data(
            self.hparams.data_dir, self.hparams.num_iters_for_train_examples_history
        )

        LOG_INFO(
            "Loaded %d examples from %s", len(self.data), str(self.hparams.data_dir)
        )

    def load_data(self, data_dir, max_iters):
        """Load data from files"""
        data = []
        for examples_fp in list(data_dir.iterdir())[-max_iters:]:
            with open(examples_fp, "rb") as f:
                data.extend(Unpickler(f).load())
        return data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return {
            "observation": torch.from_numpy(self.data[idx][0]).float(),
            "mask": torch.from_numpy(self.data[idx][1]).bool(),
            "prob": torch.from_numpy(self.data[idx][2]).float(),
            "score": torch.from_numpy(self.data[idx][3]).float(),
        }
