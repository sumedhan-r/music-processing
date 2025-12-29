"""TensorFlow base model implementation."""

from typing import Any
import tensorflow as tf

from src.ml.supervised.models.base import BaseModel


class TensorFlowBaseModel(BaseModel, tf.keras.Model):
    """Base class for TensorFlow models.

    Combines the framework-agnostic BaseModel interface with tf.keras.Model.
    """

    def __init__(self, **kwargs: Any):
        """Initialize TensorFlow base model."""
        tf.keras.Model.__init__(self, **kwargs)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the model (TensorFlow convention).

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement call()")

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass (BaseModel interface compatibility).

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.call(x, training=False)

    def get_config(self) -> dict[str, Any]:
        """Get model configuration.

        Returns:
            Dictionary containing model configuration
        """
        return {
            "model_class": self.__class__.__name__,
            "num_parameters": self.count_params(),
            "trainable_parameters": sum(
                tf.size(w).numpy() for w in self.trainable_weights
            ),
        }

    def load_weights(self, path: str) -> None:
        """Load model weights from file.

        Args:
            path: Path to TensorFlow checkpoint or SavedModel
        """
        try:
            super().load_weights(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {path}: {e}") from e

    def save_weights(self, path: str) -> None:
        """Save model weights to file.

        Args:
            path: Path to save TensorFlow checkpoint
        """
        try:
            super().save_weights(path)
        except Exception as e:
            raise RuntimeError(f"Failed to save weights to {path}: {e}") from e

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with total and trainable parameter counts
        """
        total = self.count_params()
        trainable = sum(tf.size(w).numpy() for w in self.trainable_weights)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
