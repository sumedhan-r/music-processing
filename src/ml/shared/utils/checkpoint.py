"""Repository for checkpoint management operations."""

import json
from pathlib import Path
from typing import Any

from src.ml.shared.config.schemas import ModelCheckpoint


class CheckpointRepository:
    """Manages saving and loading model checkpoints in a framework-agnostic way.

    This repository handles checkpoint metadata and organization,
    while delegating framework-specific serialization to model implementations.
    """

    def __init__(self, checkpoint_dir: str | Path):
        """Initialize checkpoint repository.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, checkpoint: ModelCheckpoint) -> None:
        """Save checkpoint metadata.

        Args:
            checkpoint: Checkpoint metadata
        """
        metadata_path = Path(checkpoint.checkpoint_path).with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(checkpoint.model_dump(), f, indent=2)

    def load_metadata(self, checkpoint_path: str | Path) -> ModelCheckpoint:
        """Load checkpoint metadata.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint metadata
        """
        metadata_path = Path(checkpoint_path).with_suffix(".json")
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return ModelCheckpoint(**data)

    def list_checkpoints(self) -> list[Path]:
        """List all checkpoint files in the directory.

        Returns:
            List of checkpoint file paths
        """
        return sorted(self.checkpoint_dir.glob("*.json"))

    def get_best_checkpoint(self, metric_name: str, mode: str = "min") -> Path | None:
        """Get the best checkpoint based on a metric.

        Args:
            metric_name: Name of metric to compare
            mode: 'min' or 'max' for optimization direction

        Returns:
            Path to best checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        best_checkpoint = None
        best_value = float("inf") if mode == "min" else float("-inf")

        for ckpt_path in checkpoints:
            metadata = self.load_metadata(ckpt_path.with_suffix(""))
            if metric_name in metadata.metrics:
                value = metadata.metrics[metric_name]
                if (mode == "min" and value < best_value) or (
                    mode == "max" and value > best_value
                ):
                    best_value = value
                    best_checkpoint = Path(metadata.checkpoint_path)

        return best_checkpoint

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """Remove old checkpoints, keeping only the most recent N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_last_n:
            return

        # Remove oldest checkpoints
        for ckpt_path in checkpoints[:-keep_last_n]:
            # Remove both metadata and model files
            ckpt_path.unlink()
            model_path = ckpt_path.with_suffix("")
            if model_path.exists():
                model_path.unlink()
