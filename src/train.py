"""Entry point for training the model."""
import os
import sys
from pathlib import Path

import gymnasium as gym
import wandb

from src import hparams
