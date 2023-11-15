"""Module with dataset classes for MCTS"""
from pickle import Unpickler

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..hparams import MCTSHparams
from ..utils import LOG_INFO


class MCTSDataset(Dataset):
    """Dataset for MCTS"""

    def __init__(self, hparams: MCTSHparams, is_train: bool = True):
        super().__init__()

        # Get hyperparameters
        self.hparams = hparams
        self.data_dir = self.hparams.data_dir if is_train else self.hparams.val_data_dir

        # Load data from files
        self.data = self.load_data(
            self.data_dir, self.hparams.num_iters_for_train_examples_history
        )

        LOG_INFO(
            "Loaded %d examples from %s", len(self.data), str(self.data_dir)
        )

    def load_data(self, data_dir, max_iters):
        """Load data from files"""
        data = []
        examples_fp_list = list(data_dir.rglob('*.examples'))
        for examples_fp in examples_fp_list[-max_iters:]:
            with open(examples_fp, "rb") as f:
                data.extend(Unpickler(f).load())
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "observation": torch.from_numpy(self.data[idx][0]).float(),
            "mask": torch.from_numpy(self.data[idx][1]).bool(),
            "prob": torch.from_numpy(self.data[idx][2]).float(),
            "score": torch.from_numpy(self.data[idx][3]).float(),
        }


def collate_dataset_fn(samples: list[dict]):
    batch: dict = {k: pad_sequence([s[k] for s in samples], batch_first=True) for k in samples[0]}
    return batch
