"""TensorFlow audio classifier model example."""

import tensorflow as tf
from typing import Any

from src.ml.supervised.models.tensorflow.base_model import TensorFlowBaseModel


class AudioClassifier(TensorFlowBaseModel):
    """Example audio classification model using TensorFlow.

    This is a simple CNN-based classifier for demonstration.
    Replace with your actual model architecture.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int = 10,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
        **kwargs: Any,
    ):
        """Initialize audio classifier.

        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            **kwargs: Additional arguments for tf.keras.Model
        """
        super().__init__(**kwargs)

        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        self.input_shape_config = input_shape
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        # Build convolutional layers
        self.conv_blocks = []
        for hidden_dim in hidden_dims:
            block = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        hidden_dim,
                        kernel_size=3,
                        padding="same",
                        activation=None,
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
                    tf.keras.layers.Dropout(dropout),
                ],
                name=f"conv_block_{hidden_dim}",
            )
            self.conv_blocks.append(block)

        # Global pooling
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

        # Classification head
        self.classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(num_classes),
            ],
            name="classifier",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether in training mode

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = inputs

        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x, training=training)

        # Global pooling
        x = self.global_pool(x)

        # Classifier
        x = self.classifier(x, training=training)

        return x

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape_config,
                "num_classes": self.num_classes,
                "hidden_dims": self.hidden_dims,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def predict_classes(self, x: tf.Tensor) -> tf.Tensor:
        """Get predictions (class indices).

        Args:
            x: Input tensor

        Returns:
            Predicted class indices
        """
        logits = self.call(x, training=False)
        predictions = tf.argmax(logits, axis=1)
        return predictions

    def predict_proba(self, x: tf.Tensor) -> tf.Tensor:
        """Get prediction probabilities.

        Args:
            x: Input tensor

        Returns:
            Class probabilities
        """
        logits = self.call(x, training=False)
        probabilities = tf.nn.softmax(logits, axis=1)
        return probabilities
