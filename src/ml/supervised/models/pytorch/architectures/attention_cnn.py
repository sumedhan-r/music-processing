"""CNN with attention mechanisms for audio classification.

Combines convolutional layers with attention for better feature learning.
"""

import torch
import torch.nn as nn

from src.ml.supervised.models.pytorch.base_model import PyTorchBaseModel
from src.ml.supervised.models.pytorch.registry import ModelRegistry
from src.ml.supervised.models.pytorch.components.conv_blocks import ConvBlock
from src.ml.supervised.models.pytorch.components.attention import ChannelAttention
from src.ml.supervised.models.pytorch.components.pooling import AttentionPooling


@ModelRegistry.register("attention_cnn")
class AttentionCNN(PyTorchBaseModel):
    """CNN with channel attention mechanisms.

    Research use: Investigate whether attention improves audio classification.

    Architecture:
        Conv -> Attention -> Conv -> Attention -> Pool -> FC

    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        hidden_dims: List of channel dimensions
        reduction_ratio: Attention reduction ratio
        dropout_rate: Dropout probability
        use_attention_pooling: Use attention pooling instead of global pooling

    Example:
        >>> model = ModelRegistry.create("attention_cnn", num_classes=10)
        >>> output = model(spectrogram)
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        hidden_dims: list = [64, 128, 256],
        reduction_ratio: int = 16,
        dropout_rate: float = 0.3,
        use_attention_pooling: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        # Build conv + attention layers
        layers = []
        in_ch = input_channels

        for out_ch in hidden_dims:
            # Convolutional block
            layers.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    activation="relu",
                    dropout_rate=dropout_rate,
                )
            )

            # Channel attention
            layers.append(
                ChannelAttention(
                    channels=out_ch,
                    reduction_ratio=reduction_ratio,
                )
            )

            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # Pooling
        if use_attention_pooling:
            self.pool = AttentionPooling(
                input_dim=hidden_dims[-1],
                hidden_dim=hidden_dims[-1] // 2,
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten(1)

        self.use_attention_pooling = use_attention_pooling

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

        if self.use_attention_pooling:
            x = self.pool(x)
        else:
            x = self.pool(x)
            x = self.flatten(x)

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
