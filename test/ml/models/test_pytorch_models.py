"""Tests for PyTorch models with mocking."""

from unittest.mock import MagicMock, Mock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pytorch]


class TestPyTorchBaseModel:
    """Test PyTorchBaseModel with mocked torch."""

    @patch("src.ml.supervised.models.pytorch.base_model.torch")
    @patch("src.ml.supervised.models.pytorch.base_model.nn")
    def test_get_config_returns_parameter_counts(self, mock_nn, mock_torch):
        """Test that get_config returns parameter information."""
        from src.ml.supervised.models.pytorch.base_model import PyTorchBaseModel

        # Create a concrete test class
        class TestModel(PyTorchBaseModel):
            def __init__(self):
                super().__init__()
                self.fc = Mock()

            def forward(self, x):
                return self.fc(x)

        model = TestModel()

        # Mock parameters
        mock_param1 = Mock()
        mock_param1.numel.return_value = 100
        mock_param1.requires_grad = True

        mock_param2 = Mock()
        mock_param2.numel.return_value = 50
        mock_param2.requires_grad = False

        model.parameters = Mock(return_value=[mock_param1, mock_param2])

        config = model.get_config()

        assert "model_class" in config
        assert config["model_class"] == "TestModel"
        assert config["num_parameters"] == 150
        assert config["trainable_parameters"] == 100

    @patch("src.ml.supervised.models.pytorch.base_model.torch")
    def test_save_weights_calls_torch_save(self, mock_torch):
        """Test that save_weights calls torch.save()."""
        from src.ml.supervised.models.pytorch.base_model import PyTorchBaseModel

        class TestModel(PyTorchBaseModel):
            def forward(self, x):
                return x

        model = TestModel()
        model.state_dict = Mock(return_value={"weights": "data"})

        model.save_weights("/path/to/model.pth")

        mock_torch.save.assert_called_once_with({"weights": "data"}, "/path/to/model.pth")

    @patch("src.ml.supervised.models.pytorch.base_model.torch")
    def test_load_weights_calls_torch_load(self, mock_torch):
        """Test that load_weights calls torch.load()."""
        from src.ml.supervised.models.pytorch.base_model import PyTorchBaseModel

        class TestModel(PyTorchBaseModel):
            def forward(self, x):
                return x

        model = TestModel()
        model.load_state_dict = Mock()
        mock_torch.load.return_value = {"weights": "data"}

        model.load_weights("/path/to/model.pth")

        mock_torch.load.assert_called_once()
        model.load_state_dict.assert_called_once_with({"weights": "data"})


class TestAudioClassifier:
    """Test AudioClassifier with mocked torch."""

    @pytest.mark.skip(reason="Requires actual PyTorch - use integration test instead")
    def test_audio_classifier_initialization(self):
        """Test AudioClassifier initialization."""
        # This would be an integration test requiring actual PyTorch
        pass

    @pytest.mark.skip(reason="Requires actual PyTorch - use integration test instead")
    def test_audio_classifier_forward_pass(self):
        """Test AudioClassifier forward pass."""
        # This would be an integration test requiring actual PyTorch
        pass


class TestCheckpointUtils:
    """Test PyTorch checkpoint utilities."""

    @patch("src.ml.supervised.models.pytorch.checkpoint_utils.torch")
    def test_save_checkpoint(self, mock_torch, tmp_path):
        """Test saving a PyTorch checkpoint."""
        from src.ml.supervised.models.pytorch.checkpoint_utils import save_checkpoint

        mock_model = Mock()
        mock_model.state_dict.return_value = {"model": "state"}
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {"optimizer": "state"}
        mock_model.get_config.return_value = {"config": "data"}

        checkpoint_path = tmp_path / "checkpoint.pth"

        save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=10,
            metrics={"loss": 0.5},
            checkpoint_path=checkpoint_path,
        )

        # Verify torch.save was called
        mock_torch.save.assert_called_once()
        call_args = mock_torch.save.call_args[0]
        checkpoint_data = call_args[0]

        assert checkpoint_data["epoch"] == 10
        assert checkpoint_data["metrics"] == {"loss": 0.5}
        assert checkpoint_data["model_state_dict"] == {"model": "state"}
        assert checkpoint_data["optimizer_state_dict"] == {"optimizer": "state"}

    @patch("src.ml.supervised.models.pytorch.checkpoint_utils.torch")
    def test_load_checkpoint(self, mock_torch, tmp_path):
        """Test loading a PyTorch checkpoint."""
        from src.ml.supervised.models.pytorch.checkpoint_utils import load_checkpoint

        mock_model = Mock()
        mock_optimizer = Mock()

        # Mock the loaded checkpoint data
        mock_torch.load.return_value = {
            "epoch": 10,
            "model_state_dict": {"model": "state"},
            "optimizer_state_dict": {"optimizer": "state"},
            "metrics": {"loss": 0.5},
            "model_config": {"config": "data"},
        }

        checkpoint_path = tmp_path / "checkpoint.pth"

        result = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=mock_model,
            optimizer=mock_optimizer,
        )

        # Verify torch.load was called
        mock_torch.load.assert_called_once()

        # Verify model and optimizer state dicts were loaded
        mock_model.load_state_dict.assert_called_once_with({"model": "state"})
        mock_optimizer.load_state_dict.assert_called_once_with({"optimizer": "state"})

        # Verify returned data
        assert result["epoch"] == 10
        assert result["metrics"] == {"loss": 0.5}
        assert result["model_config"] == {"config": "data"}
