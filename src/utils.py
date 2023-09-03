"""Utility functions for the project."""
import random
import logging

import torch
import numpy as np
import coloredlogs

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(
    coloredlogs.ColoredFormatter("%(asctime)s %(levelname)s %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def LOG_INFO(msg: str):
    """Log an info message."""
    logger.info(msg)


def LOG_WARNING(msg: str):
    """Log a warning message."""
    logger.warning(msg)


def LOG_ERROR(msg: str):
    """Log an error message."""
    logger.error(msg)


def seed(seed_number: int):
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
