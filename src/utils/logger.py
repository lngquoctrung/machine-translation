"""Logging utilities"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_training_start(logger, config):
    """Log training start info"""
    logger.info("=" * 60)
    logger.info("TRAINING STARTED")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def log_training_end(logger, history, elapsed_time):
    """Log training end info"""
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed_time / 3600:.2f} hours")
    logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
    logger.info(f"Final val_loss: {history.history['val_loss'][-1]:.4f}")
    logger.info("=" * 60)
