"""Tests for WandbLogger connector.

These are integration-style tests that mock the wandb library
to test that the connector correctly uses the wandb Run API.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]


class TestWandbLogger:
    """Test WandbLogger with mocked wandb library."""

    @patch("src.ml.shared.logging.wandb.wandb")
    def test_init_creates_run(self, mock_wandb):
        """Test that WandbLogger initializes a wandb run."""
        from src.ml.shared.logging.wandb import WandbLogger

        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(project="test-project", name="test-run")

        mock_wandb.init.assert_called_once_with(
            project="test-project",
            name="test-run",
            config=None,
            tags=None,
            notes=None,
        )
        assert logger.run == mock_run

    @patch("src.ml.shared.logging.wandb.wandb")
    def test_log_metric_uses_run(self, mock_wandb):
        """Test that log_metric calls run.log()."""
        from src.ml.shared.logging.wandb import WandbLogger

        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(project="test-project")
        logger.log_metric("loss", 0.5, step=10)

        mock_run.log.assert_called_once_with({"loss": 0.5}, step=10)

    @patch("src.ml.shared.logging.wandb.wandb")
    def test_log_metrics_uses_run(self, mock_wandb):
        """Test that log_metrics calls run.log() with multiple metrics."""
        from src.ml.shared.logging.wandb import WandbLogger

        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(project="test-project")
        metrics = {"loss": 0.5, "accuracy": 0.95}
        logger.log_metrics(metrics, step=10)

        mock_run.log.assert_called_once_with(metrics, step=10)

    @patch("src.ml.shared.logging.wandb.wandb")
    def test_log_params_uses_run_config(self, mock_wandb):
        """Test that log_params calls run.config.update()."""
        from src.ml.shared.logging.wandb import WandbLogger

        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(project="test-project")
        params = {"learning_rate": 0.001, "batch_size": 32}
        logger.log_params(params)

        mock_run.config.update.assert_called_once_with(params)

    @patch("src.ml.shared.logging.wandb.wandb")
    def test_log_artifact_uses_run(self, mock_wandb, temp_artifact_file):
        """Test that log_artifact calls run.log_artifact()."""
        from src.ml.shared.logging.wandb import WandbLogger

        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        logger = WandbLogger(project="test-project")
        logger.log_artifact(temp_artifact_file, artifact_type="model")

        # Check that Artifact was created
        mock_wandb.Artifact.assert_called_once_with(
            name=temp_artifact_file.stem, type="model"
        )
        # Check that file was added to artifact
        mock_artifact.add_file.assert_called_once_with(str(temp_artifact_file))
        # Check that artifact was logged
        mock_run.log_artifact.assert_called_once_with(mock_artifact)

    @patch("src.ml.shared.logging.wandb.wandb")
    def test_log_table_uses_run(self, mock_wandb):
        """Test that log_table calls run.log() with table data."""
        from src.ml.shared.logging.wandb import WandbLogger

        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table

        logger = WandbLogger(project="test-project")
        logger.log_table("predictions", mock_table)

        mock_run.log.assert_called_once_with({"predictions": mock_table})

    @patch("src.ml.shared.logging.wandb.wandb")
    def test_save_file_uses_run(self, mock_wandb, temp_artifact_file):
        """Test that save_file calls run.save()."""
        from src.ml.shared.logging.wandb import WandbLogger

        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(project="test-project")
        logger.save_file(temp_artifact_file)

        mock_run.save.assert_called_once_with(str(temp_artifact_file))

    @patch("src.ml.shared.logging.wandb.wandb")
    def test_finish_uses_run(self, mock_wandb):
        """Test that finish calls run.finish()."""
        from src.ml.shared.logging.wandb import WandbLogger

        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        logger = WandbLogger(project="test-project")
        logger.finish()

        mock_run.finish.assert_called_once()

    def test_import_error_when_wandb_not_installed(self):
        """Test that ImportError is raised when wandb is not installed."""
        from src.ml.shared.logging.wandb import WandbLogger

        with patch.dict("sys.modules", {"wandb": None}):
            with pytest.raises(ImportError) as exc_info:
                # This will fail during import, not init
                import importlib
                import src.ml.shared.logging.wandb
                importlib.reload(src.ml.shared.logging.wandb)
