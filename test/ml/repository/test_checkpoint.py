"""Tests for checkpoint repository."""

import json
from pathlib import Path

import pytest

from src.ml.shared.utils.checkpoint import CheckpointRepository
from src.ml.shared.config.schemas import ModelCheckpoint

pytestmark = pytest.mark.unit


class TestCheckpointRepository:
    """Test CheckpointRepository."""

    def test_init_creates_directory(self, tmp_path):
        """Test that CheckpointRepository creates checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        repo = CheckpointRepository(checkpoint_dir)

        assert repo.checkpoint_dir.exists()
        assert repo.checkpoint_dir.is_dir()

    def test_save_metadata(self, tmp_path, sample_checkpoint_metadata):
        """Test saving checkpoint metadata."""
        repo = CheckpointRepository(tmp_path)
        repo.save_metadata(sample_checkpoint_metadata)

        metadata_path = tmp_path / "checkpoint.json"
        assert metadata_path.exists()

        # Verify content
        with open(metadata_path, "r") as f:
            data = json.load(f)
        assert data["epoch"] == 5
        assert data["step"] == 1000
        assert data["metrics"]["accuracy"] == 0.95

    def test_load_metadata(self, tmp_path, sample_checkpoint_metadata):
        """Test loading checkpoint metadata."""
        repo = CheckpointRepository(tmp_path)
        repo.save_metadata(sample_checkpoint_metadata)

        loaded = repo.load_metadata(tmp_path / "checkpoint")

        assert loaded.epoch == sample_checkpoint_metadata.epoch
        assert loaded.step == sample_checkpoint_metadata.step
        assert loaded.metrics == sample_checkpoint_metadata.metrics

    def test_list_checkpoints(self, tmp_path):
        """Test listing checkpoints."""
        repo = CheckpointRepository(tmp_path)

        # Create multiple checkpoints
        for i in range(3):
            checkpoint = ModelCheckpoint(
                checkpoint_path=str(tmp_path / f"checkpoint_{i}.pth"),
                epoch=i,
                step=i * 100,
                metrics={"loss": 0.5 - i * 0.1},
                timestamp=f"2024-01-0{i+1}T00:00:00Z",
                config={},
            )
            repo.save_metadata(checkpoint)

        checkpoints = repo.list_checkpoints()
        assert len(checkpoints) == 3

    def test_get_best_checkpoint_min_mode(self, tmp_path):
        """Test getting best checkpoint in min mode."""
        repo = CheckpointRepository(tmp_path)

        # Create checkpoints with different losses
        for i, loss in enumerate([0.5, 0.3, 0.4]):
            checkpoint = ModelCheckpoint(
                checkpoint_path=str(tmp_path / f"checkpoint_{i}.pth"),
                epoch=i,
                step=i * 100,
                metrics={"loss": loss, "accuracy": 0.9},
                timestamp=f"2024-01-0{i+1}T00:00:00Z",
                config={},
            )
            repo.save_metadata(checkpoint)

        best = repo.get_best_checkpoint("loss", mode="min")
        assert best == tmp_path / "checkpoint_1.pth"  # Index 1 has loss=0.3

    def test_get_best_checkpoint_max_mode(self, tmp_path):
        """Test getting best checkpoint in max mode."""
        repo = CheckpointRepository(tmp_path)

        # Create checkpoints with different accuracies
        for i, acc in enumerate([0.85, 0.92, 0.88]):
            checkpoint = ModelCheckpoint(
                checkpoint_path=str(tmp_path / f"checkpoint_{i}.pth"),
                epoch=i,
                step=i * 100,
                metrics={"loss": 0.3, "accuracy": acc},
                timestamp=f"2024-01-0{i+1}T00:00:00Z",
                config={},
            )
            repo.save_metadata(checkpoint)

        best = repo.get_best_checkpoint("accuracy", mode="max")
        assert best == tmp_path / "checkpoint_1.pth"  # Index 1 has accuracy=0.92

    def test_get_best_checkpoint_no_checkpoints(self, tmp_path):
        """Test getting best checkpoint when no checkpoints exist."""
        repo = CheckpointRepository(tmp_path)
        best = repo.get_best_checkpoint("loss", mode="min")
        assert best is None

    def test_cleanup_old_checkpoints(self, tmp_path):
        """Test cleanup of old checkpoints."""
        repo = CheckpointRepository(tmp_path)

        # Create 10 checkpoints
        for i in range(10):
            checkpoint = ModelCheckpoint(
                checkpoint_path=str(tmp_path / f"checkpoint_{i}.pth"),
                epoch=i,
                step=i * 100,
                metrics={"loss": 0.5},
                timestamp=f"2024-01-0{i+1}T00:00:00Z",
                config={},
            )
            repo.save_metadata(checkpoint)
            # Create dummy model file
            (tmp_path / f"checkpoint_{i}.pth").touch()

        # Keep only last 5
        repo.cleanup_old_checkpoints(keep_last_n=5)

        remaining = repo.list_checkpoints()
        assert len(remaining) == 5

        # Check that the newest ones are kept
        assert (tmp_path / "checkpoint_9.json").exists()
        assert not (tmp_path / "checkpoint_0.json").exists()
