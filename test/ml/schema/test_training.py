"""Tests for training configuration schemas."""

import pytest
from pydantic import ValidationError

from src.ml.shared.config.schemas import (
    ExperimentConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)

pytestmark = pytest.mark.unit


class TestOptimizerConfig:
    """Test OptimizerConfig schema."""

    def test_optimizer_config_valid_adam(self):
        """Test OptimizerConfig with valid Adam configuration."""
        config = OptimizerConfig(
            name="adam",
            learning_rate=0.001,
            weight_decay=0.0001,
            betas=(0.9, 0.999),
        )

        assert config.name == "adam"
        assert config.learning_rate == 0.001
        assert config.weight_decay == 0.0001
        assert config.betas == (0.9, 0.999)

    def test_optimizer_config_valid_sgd(self):
        """Test OptimizerConfig with valid SGD configuration."""
        config = OptimizerConfig(
            name="sgd",
            learning_rate=0.01,
            momentum=0.9,
        )

        assert config.name == "sgd"
        assert config.learning_rate == 0.01
        assert config.momentum == 0.9

    def test_optimizer_config_negative_learning_rate(self):
        """Test OptimizerConfig rejects negative learning rate."""
        with pytest.raises(ValidationError) as exc_info:
            OptimizerConfig(
                name="adam",
                learning_rate=-0.001,
            )
        assert "Input should be greater than 0" in str(exc_info.value)

    def test_optimizer_config_invalid_name(self):
        """Test OptimizerConfig rejects invalid optimizer name."""
        with pytest.raises(ValidationError) as exc_info:
            OptimizerConfig(
                name="invalid_optimizer",  # type: ignore[arg-type]
                learning_rate=0.001,
            )
        assert "Input should be" in str(exc_info.value)

    def test_optimizer_config_negative_weight_decay(self):
        """Test OptimizerConfig rejects negative weight decay."""
        with pytest.raises(ValidationError) as exc_info:
            OptimizerConfig(
                name="adam",
                learning_rate=0.001,
                weight_decay=-0.1,
            )
        assert "Input should be greater than or equal to 0" in str(exc_info.value)


class TestSchedulerConfig:
    """Test SchedulerConfig schema."""

    def test_scheduler_config_valid_step(self):
        """Test SchedulerConfig with valid StepLR configuration."""
        config = SchedulerConfig(
            name="step",
            step_size=10,
            gamma=0.1,
        )

        assert config.name == "step"
        assert config.step_size == 10
        assert config.gamma == 0.1

    def test_scheduler_config_none(self):
        """Test SchedulerConfig with no scheduler."""
        config = SchedulerConfig(name="none")

        assert config.name == "none"
        assert config.step_size is None
        assert config.gamma is None

    def test_scheduler_config_invalid_name(self):
        """Test SchedulerConfig rejects invalid scheduler name."""
        with pytest.raises(ValidationError):
            SchedulerConfig(name="invalid_scheduler")  # type: ignore[arg-type]


class TestTrainingConfig:
    """Test TrainingConfig schema."""

    def test_training_config_valid_data(self):
        """Test TrainingConfig with valid data."""
        optimizer = OptimizerConfig(name="adam", learning_rate=0.001)
        scheduler = SchedulerConfig(name="step", step_size=10, gamma=0.1)

        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_value=1.0,
            early_stopping_patience=10,
        )

        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.optimizer.name == "adam"
        assert config.scheduler.name == "step"  # type: ignore[union-attr]
        assert config.gradient_clip_value == 1.0
        assert config.early_stopping_patience == 10

    def test_training_config_negative_epochs(self):
        """Test TrainingConfig rejects negative epochs."""
        optimizer = OptimizerConfig(name="adam", learning_rate=0.001)

        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                epochs=-1,
                batch_size=32,
                optimizer=optimizer,
            )
        assert "Input should be greater than 0" in str(exc_info.value)

    def test_training_config_zero_batch_size(self):
        """Test TrainingConfig rejects zero batch size."""
        optimizer = OptimizerConfig(name="adam", learning_rate=0.001)

        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                epochs=100,
                batch_size=0,
                optimizer=optimizer,
            )
        assert "Input should be greater than 0" in str(exc_info.value)


class TestLoggerConfig:
    """Test LoggerConfig schema."""

    def test_logger_config_wandb(self):
        """Test LoggerConfig with wandb provider."""
        config = LoggerConfig(
            provider="wandb",
            project_name="music-processing",
            experiment_name="exp-001",
            tags=["test", "audio"],
            notes="Test experiment",
        )

        assert config.provider == "wandb"
        assert config.project_name == "music-processing"
        assert config.experiment_name == "exp-001"
        assert config.tags == ["test", "audio"]
        assert config.notes == "Test experiment"

    def test_logger_config_mlflow(self):
        """Test LoggerConfig with mlflow provider."""
        config = LoggerConfig(
            provider="mlflow",
            project_name="music-processing",
            experiment_name="exp-001",
        )

        assert config.provider == "mlflow"

    def test_logger_config_invalid_provider(self):
        """Test LoggerConfig rejects invalid provider."""
        with pytest.raises(ValidationError):
            LoggerConfig(
                provider="invalid_logger",  # type: ignore[arg-type]
                project_name="music-processing",
                experiment_name="exp-001",
            )


class TestExperimentConfig:
    """Test ExperimentConfig schema."""

    def test_experiment_config_valid(self):
        """Test ExperimentConfig with valid data."""
        optimizer = OptimizerConfig(name="adam", learning_rate=0.001)
        training = TrainingConfig(
            epochs=100,
            batch_size=32,
            optimizer=optimizer,
        )
        logger = LoggerConfig(
            provider="wandb",
            project_name="music-processing",
            experiment_name="exp-001",
        )

        config = ExperimentConfig(
            experiment_id="exp-001",
            training=training,
            logger=logger,
            model_config={"hidden_dims": [64, 128]},
            data_config={"data_dir": "/path/to/data"},
            seed=42,
        )

        assert config.experiment_id == "exp-001"
        assert config.training.epochs == 100
        assert config.logger.provider == "wandb"
        assert config.model_config["hidden_dims"] == [64, 128]
        assert config.seed == 42

    def test_experiment_config_missing_required_fields(self):
        """Test ExperimentConfig with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig()  # type: ignore[call-arg]
        assert "Field required" in str(exc_info.value)
