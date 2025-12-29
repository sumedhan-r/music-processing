"""TensorFlow-specific checkpoint utilities."""

import tensorflow as tf
from pathlib import Path
from typing import Any

from src.ml.supervised.models.tensorflow.base_model import TensorFlowBaseModel


def save_checkpoint(
    model: TensorFlowBaseModel,
    optimizer: tf.keras.optimizers.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    checkpoint_path: str | Path,
) -> None:
    """Save a complete training checkpoint.

    Args:
        model: TensorFlow model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_path: Path to save checkpoint (without extension)
    """
    checkpoint_path = Path(checkpoint_path)

    # Save model weights
    model.save_weights(str(checkpoint_path / "model"))

    # Save optimizer state
    with open(checkpoint_path / "optimizer.pkl", "wb") as f:
        import pickle

        pickle.dump(
            {
                "optimizer_config": optimizer.get_config(),
                "optimizer_weights": optimizer.get_weights(),
            },
            f,
        )

    # Save metadata
    import json

    metadata = {
        "epoch": epoch,
        "metrics": metrics,
        "model_config": model.get_config(),
    }
    with open(checkpoint_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: TensorFlowBaseModel,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load weights into
        optimizer: Optional optimizer to restore state

    Returns:
        Dictionary containing epoch, metrics, and config
    """
    checkpoint_path = Path(checkpoint_path)

    # Load model weights
    model.load_weights(str(checkpoint_path / "model"))

    # Load optimizer state if provided
    if optimizer is not None:
        import pickle

        with open(checkpoint_path / "optimizer.pkl", "rb") as f:
            optimizer_data = pickle.load(f)
            optimizer.set_weights(optimizer_data["optimizer_weights"])

    # Load metadata
    import json

    with open(checkpoint_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    return {
        "epoch": metadata["epoch"],
        "metrics": metadata["metrics"],
        "model_config": metadata.get("model_config", {}),
    }


def export_to_saved_model(
    model: TensorFlowBaseModel, output_path: str | Path, **kwargs: Any
) -> None:
    """Export TensorFlow model to SavedModel format.

    Args:
        model: TensorFlow model
        output_path: Path to save SavedModel
        **kwargs: Additional arguments for model.save()
    """
    model.save(str(output_path), save_format="tf", **kwargs)


def export_to_tflite(
    model: TensorFlowBaseModel, output_path: str | Path, **kwargs: Any
) -> None:
    """Export TensorFlow model to TFLite format.

    Args:
        model: TensorFlow model
        output_path: Path to save TFLite file
        **kwargs: Additional converter options
    """
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply optimizations if specified
    if kwargs.get("optimize", False):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save the model
    with open(output_path, "wb") as f:
        f.write(tflite_model)
