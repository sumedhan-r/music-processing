"""Reusable training utility functions."""

import random
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    # TODO: Add numpy and framework-specific seeding
    # import numpy as np
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # etc.


def save_checkpoint(
    model: Any,
    optimizer: Any,
    epoch: int,
    metrics: dict[str, float],
    checkpoint_dir: str | Path,
) -> str:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoint

    Returns:
        Path to saved checkpoint
    """
    # Placeholder implementation
    raise NotImplementedError("Implement checkpoint saving based on your ML framework")


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary containing model, optimizer, epoch, and metrics
    """
    # Placeholder implementation
    raise NotImplementedError("Implement checkpoint loading based on your ML framework")


def early_stopping_check(
    metrics_history: list[float], patience: int, mode: str = "min"
) -> bool:
    """Check if early stopping criteria is met.

    Args:
        metrics_history: History of metric values
        patience: Number of epochs to wait for improvement
        mode: 'min' for metrics that should decrease, 'max' for metrics that should increase

    Returns:
        True if training should stop, False otherwise
    """
    if len(metrics_history) < patience + 1:
        return False

    recent = metrics_history[-patience:]
    best = metrics_history[:-patience]

    if mode == "min":
        return min(recent) >= min(best)
    else:
        return max(recent) <= max(best)
