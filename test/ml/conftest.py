"""Shared fixtures for ML tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest


# ==================== Pytest Markers ====================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "pytorch: PyTorch-specific tests")
    config.addinivalue_line("markers", "tensorflow: TensorFlow-specific tests")
    config.addinivalue_line(
        "markers", "requires_wandb: Tests that require wandb credentials"
    )
    config.addinivalue_line(
        "markers", "requires_mlflow: Tests that require mlflow setup"
    )


# ==================== Test Data Fixtures ====================


@pytest.fixture(scope="session")
def test_config_data():
    """Test configuration data for experiments."""
    return {
        "experiment_id": "test-exp-001",
        "project_name": "test-music-processing",
        "experiment_name": "test-run-001",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "num_classes": 10,
        "input_channels": 1,
        "hidden_dims": [64, 128, 256],
        "dropout": 0.5,
    }


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_config_file(tmp_path, test_config_data):
    """Temporary YAML config file."""
    import yaml

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(test_config_data, f)
    return config_file


@pytest.fixture
def temp_artifact_file(tmp_path):
    """Temporary artifact file for testing."""
    artifact_file = tmp_path / "model.pth"
    artifact_file.write_text("fake model data")
    return artifact_file


# ==================== Mock Logger Fixtures ====================


@pytest.fixture
def mock_wandb():
    """Mock wandb module for testing without actual API calls."""
    mock = MagicMock()
    mock_run = MagicMock()
    mock_run.id = "test-run-123"
    mock_run.name = "test-experiment"
    mock.init.return_value = mock_run
    mock.Table = MagicMock
    mock.Artifact = MagicMock
    return mock


@pytest.fixture
def mock_mlflow():
    """Mock mlflow module for testing without actual API calls."""
    mock = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-mlflow-run-123"
    mock.start_run.return_value = mock_run
    mock.active_run.return_value = mock_run
    return mock


@pytest.fixture
def mock_tensorboard_writer():
    """Mock TensorBoard SummaryWriter."""
    mock = MagicMock()
    return mock


# ==================== PyTorch Fixtures ====================


@pytest.fixture
def mock_torch():
    """Mock PyTorch module for unit testing."""
    mock = MagicMock()

    # Mock tensor
    mock_tensor = MagicMock()
    mock_tensor.shape = (32, 1, 28, 28)
    mock_tensor.device = "cpu"
    mock.Tensor.return_value = mock_tensor
    mock.tensor.return_value = mock_tensor

    # Mock device
    mock.device.return_value = "cpu"

    # Mock save/load
    mock.save = MagicMock()
    mock.load = MagicMock(return_value={"state_dict": "fake"})

    # Mock nn.Module
    mock.nn = MagicMock()
    mock.nn.Module = type("MockModule", (), {})

    return mock


@pytest.fixture
def sample_pytorch_batch():
    """Sample PyTorch batch data for testing."""
    try:
        import torch

        batch = {
            "features": torch.randn(8, 1, 64, 64),
            "labels": torch.randint(0, 10, (8,)),
        }
        return batch
    except ImportError:
        pytest.skip("PyTorch not installed")


# ==================== TensorFlow Fixtures ====================


@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow module for unit testing."""
    mock = MagicMock()

    # Mock tensor
    mock_tensor = MagicMock()
    mock_tensor.shape = (32, 28, 28, 1)
    mock.constant.return_value = mock_tensor

    # Mock keras Model
    mock.keras = MagicMock()
    mock.keras.Model = type("MockModel", (), {})

    return mock


@pytest.fixture
def sample_tensorflow_batch():
    """Sample TensorFlow batch data for testing."""
    try:
        import tensorflow as tf

        batch = {
            "features": tf.random.normal((8, 64, 64, 1)),
            "labels": tf.random.uniform((8,), minval=0, maxval=10, dtype=tf.int32),
        }
        return batch
    except ImportError:
        pytest.skip("TensorFlow not installed")


# ==================== Model Fixtures ====================


@pytest.fixture
def sample_model_config():
    """Sample model configuration."""
    return {
        "model_name": "test-audio-classifier",
        "model_type": "cnn",
        "input_shape": (1, 64, 64),
        "output_shape": (10,),
        "num_classes": 10,
        "dropout_rate": 0.5,
        "use_batch_norm": True,
    }


@pytest.fixture
def sample_checkpoint_metadata(temp_checkpoint_dir):
    """Sample checkpoint metadata."""
    from src.ml.shared.config.schemas import ModelCheckpoint

    return ModelCheckpoint(
        checkpoint_path=str(temp_checkpoint_dir / "checkpoint.pth"),
        epoch=5,
        step=1000,
        metrics={"loss": 0.25, "accuracy": 0.95},
        timestamp="2024-01-01T00:00:00Z",
        config={"learning_rate": 0.001, "batch_size": 32},
    )


# ==================== Integration Test Fixtures ====================


@pytest.fixture(scope="session")
def integration_test_model():
    """Create a simple real model for integration testing.

    Returns either PyTorch or TensorFlow model based on availability.
    """
    try:
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)

            def forward(self, x):
                return self.fc(x)

        return SimpleModel()
    except ImportError:
        pass

    try:
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(10,))
        ])
        return model
    except ImportError:
        pytest.skip("Neither PyTorch nor TensorFlow installed")


@pytest.fixture
def integration_test_data():
    """Create simple test data for integration tests."""
    try:
        import torch
        return {
            "train": torch.randn(100, 10),
            "labels": torch.randint(0, 2, (100,)),
        }
    except ImportError:
        pass

    try:
        import tensorflow as tf
        return {
            "train": tf.random.normal((100, 10)),
            "labels": tf.random.uniform((100,), minval=0, maxval=2, dtype=tf.int32),
        }
    except ImportError:
        pytest.skip("Neither PyTorch nor TensorFlow installed")
