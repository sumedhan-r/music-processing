"""Simple CNN architecture for audio classification.

Provides a baseline CNN model for experimentation.
"""

import torch
import torch.nn as nn

from src.ml.supervised.models.pytorch.base_model import PyTorchBaseModel
from src.ml.supervised.models.pytorch.registry import ModelRegistry
from src.ml.supervised.models.pytorch.components.conv_blocks import ConvBlock
from src.ml.supervised.models.pytorch.components.pooling import AdaptivePooling


@ModelRegistry.register("simple_cnn")
class SimpleCNN(PyTorchBaseModel):
    """Simple CNN for audio classification.

    Research use: Baseline model for audio/spectrogram classification.

    Architecture:
        Conv (1->64) -> Conv (64->128) -> Conv (128->256) -> Pool -> FC

    Args:
        input_channels: Number of input channels (1 for mono spectrograms)
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability
        activation: Activation function

    Example:
        >>> # Via registry
        >>> model = ModelRegistry.create("simple_cnn", num_classes=10)

        >>> # Direct instantiation
        >>> model = SimpleCNN(input_channels=1, num_classes=10)
        >>> output = model(spectrogram)  # (B, 1, H, W) -> (B, 10)
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        hidden_dims: list = [64, 128, 256],
        dropout_rate: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        # Build conv layers
        layers = []

        in_ch = input_channels
        for out_ch in hidden_dims:
            layers.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    activation=activation,
                    dropout_rate=dropout_rate,
                )
            )
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # Global pooling
        self.pool = AdaptivePooling(output_size=1, pool_type="avg", flatten=True)

        # Classifier
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output logits (B, num_classes)
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "model_class": self.__class__.__name__,
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


@ModelRegistry.register("deep_cnn")
class DeepCNN(PyTorchBaseModel):
    """Deeper CNN with more capacity.

    Research use: When SimpleCNN underfits.

    Architecture:
        Conv blocks with increasing channels -> Pool -> FC

    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        base_channels: Base number of channels (doubles each block)
        num_blocks: Number of convolutional blocks
        dropout_rate: Dropout probability

    Example:
        >>> model = ModelRegistry.create("deep_cnn", num_classes=50, num_blocks=5)
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Build blocks with increasing channels
        layers = []
        in_ch = input_channels

        for i in range(num_blocks):
            out_ch = base_channels * (2 ** i)
            layers.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    activation="relu",
                    dropout_rate=dropout_rate,
                )
            )
            # Add max pooling after each conv block
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # Final channels after all blocks
        final_channels = base_channels * (2 ** (num_blocks - 1))

        # Global pooling
        self.pool = AdaptivePooling(output_size=1, pool_type="avg", flatten=True)

        # Classifier
        self.classifier = nn.Linear(final_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "model_class": self.__class__.__name__,
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
