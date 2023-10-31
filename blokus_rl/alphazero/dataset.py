"""Module with dataset classes for MCTS"""

import torch
from torch.utils.data import Dataset


class MCTSDataset(Dataset):
    """Dataset for MCTS"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return {
            'observation': torch.from_numpy(self.data[idx][0]).float(),
            'mask': torch.from_numpy(self.data[idx][1]).bool(),
            'prob': torch.from_numpy(self.data[idx][2]).float(),
            'score': torch.from_numpy(self.data[idx][3]).float(),
        }
