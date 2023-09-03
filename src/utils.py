"""Utility functions for the project."""
import logging

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
