"""Example supervised learning experiment script.

This script demonstrates how to use the reorganized codebase to train
a supervised learning model with experiment tracking.
"""

from pathlib import Path
from typing import Any

from src.ml.shared.config.schemas import (
    ExperimentConfig,
    LoggerConfig,
    TrainingConfig,
    OptimizerConfig,
    DatasetConfig,
)
from src.ml.shared.logging.factory import LoggerFactory
from src.ml.supervised.training.train import train_model


def create_experiment_config() -> ExperimentConfig:
    """Create a sample experiment configuration.

    Returns:
        Complete experiment configuration
    """
    # Logger configuration
    logger_config = LoggerConfig(
        provider="tensorboard",  # or "wandb", "mlflow"
        project_name="music-processing",
        experiment_name="audio-classification-exp1",
        tags=["audio", "classification", "pytorch"],
        notes="Example supervised learning experiment",
    )

    # Training configuration
    training_config = TrainingConfig(
        epochs=50,
        batch_size=32,
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=0.001,
            weight_decay=1e-5,
        ),
        gradient_clip_value=1.0,
        early_stopping_patience=10,
        validation_frequency=1,
        checkpoint_frequency=5,
    )

    # Dataset configuration (as dict, specific to your data)
    data_config = {
        "name": "audio_dataset",
        "data_dir": "data/audio",
        "sample_rate": 22050,
        "duration": 3.0,
    }

    # Model configuration (as dict, specific to your model)
    model_config = {
        "model_type": "audio_classifier",
        "num_classes": 10,
        "dropout_rate": 0.3,
    }

    return ExperimentConfig(
        experiment_id="audio-classification-exp1",
        training=training_config,
        logger=logger_config,
        model_config=model_config,
        data_config=data_config,
        seed=42,
    )


def main():
    """Run the supervised learning experiment."""
    # 1. Create experiment configuration
    config = create_experiment_config()

    # 2. Initialize logger using factory
    logger = LoggerFactory.create(config.logger)

    # 3. Log experiment parameters
    logger.log_params(config.model_dump())

    # 4. TODO: Initialize your model
    # from src.ml.supervised.models.pytorch.audio_classifier import AudioClassifier
    # model = AudioClassifier(config.model_config)

    # 5. TODO: Create data loaders
    # from src.ml.supervised.data.audio_dataset import create_dataloader, AudioDataset
    # train_dataset = AudioDataset(config.data_config["data_dir"], split="train")
    # train_loader = create_dataloader(train_dataset, config.training.batch_size)
    # val_loader = create_dataloader(val_dataset, config.training.batch_size, shuffle=False)

    # 6. TODO: Train the model
    # train_model(
    #     config=config,
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     logger=logger,
    # )

    # 7. Finish logging
    logger.finish()

    print("✓ Experiment completed!")


if __name__ == "__main__":
    main()
