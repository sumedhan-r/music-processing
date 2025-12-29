"""Model builder for creating models from configuration.

Allows building custom architectures from config dictionaries,
making it easy to experiment with different model designs.
"""

from typing import Any, Dict, List
import torch.nn as nn

from src.ml.supervised.models.pytorch.components.conv_blocks import (
    ConvBlock,
    ResBlock,
    DepthwiseSeparableConv,
)
from src.ml.supervised.models.pytorch.components.attention import (
    SelfAttention,
    ChannelAttention,
)
from src.ml.supervised.models.pytorch.components.pooling import (
    AdaptivePooling,
    AttentionPooling,
    GlobalPooling,
)


class ModelBuilder:
    """Build PyTorch models from configuration dictionaries.

    Research use: Design architectures via config for easy experimentation.

    Example:
        >>> config = {
        ...     "blocks": [
        ...         {"type": "conv", "in_channels": 1, "out_channels": 64},
        ...         {"type": "conv", "in_channels": 64, "out_channels": 128"},
        ...         {"type": "attention", "channels": 128},
        ...         {"type": "pool", "pool_type": "avg"},
        ...     ],
        ...     "classifier": {"in_features": 128, "num_classes": 10}
        ... }
        >>> model = ModelBuilder.from_config(config)
    """

    @staticmethod
    def from_config(config: Dict[str, Any]) -> nn.Module:
        """Build a model from configuration dictionary.

        Args:
            config: Configuration dictionary with 'blocks' and optionally 'classifier'

        Returns:
            Built PyTorch model

        Config format:
            {
                "blocks": [
                    {"type": "conv", "in_channels": 1, "out_channels": 64, ...},
                    {"type": "resblock", "channels": 64, ...},
                    {"type": "attention", "embed_dim": 64, ...},
                    {"type": "pool", "output_size": 1, ...},
                ],
                "classifier": {"in_features": 64, "num_classes": 10}  # Optional
            }
        """
        blocks = []

        # Build feature extraction blocks
        for block_config in config.get("blocks", []):
            block = ModelBuilder._build_block(block_config)
            if block is not None:
                blocks.append(block)

        # Create sequential model
        feature_extractor = nn.Sequential(*blocks)

        # Add classifier if specified
        if "classifier" in config:
            classifier_config = config["classifier"]
            classifier = nn.Linear(
                classifier_config["in_features"],
                classifier_config["num_classes"],
            )

            # Combine feature extractor and classifier
            class ConfigurableModel(nn.Module):
                def __init__(self, features, classifier):
                    super().__init__()
                    self.features = features
                    self.classifier = classifier

                def forward(self, x):
                    x = self.features(x)
                    x = x.flatten(1)
                    x = self.classifier(x)
                    return x

            return ConfigurableModel(feature_extractor, classifier)

        # Return just feature extractor
        return feature_extractor

    @staticmethod
    def _build_block(config: Dict[str, Any]) -> nn.Module:
        """Build a single block from configuration.

        Args:
            config: Block configuration dictionary

        Returns:
            Built block module
        """
        block_type = config["type"].lower()

        # Convolutional blocks
        if block_type == "conv":
            return ConvBlock(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                kernel_size=config.get("kernel_size", 3),
                stride=config.get("stride", 1),
                activation=config.get("activation", "relu"),
                use_batch_norm=config.get("use_batch_norm", True),
                dropout_rate=config.get("dropout_rate", 0.0),
            )

        elif block_type == "resblock":
            return ResBlock(
                channels=config["channels"],
                kernel_size=config.get("kernel_size", 3),
                activation=config.get("activation", "relu"),
                use_batch_norm=config.get("use_batch_norm", True),
            )

        elif block_type == "depthwise_separable":
            return DepthwiseSeparableConv(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                kernel_size=config.get("kernel_size", 3),
                stride=config.get("stride", 1),
                activation=config.get("activation", "relu"),
            )

        # Attention blocks
        elif block_type == "self_attention":
            return SelfAttention(
                embed_dim=config["embed_dim"],
                num_heads=config.get("num_heads", 1),
                dropout=config.get("dropout", 0.1),
            )

        elif block_type == "channel_attention":
            return ChannelAttention(
                channels=config["channels"],
                reduction_ratio=config.get("reduction_ratio", 16),
            )

        # Pooling blocks
        elif block_type == "pool":
            return AdaptivePooling(
                output_size=config.get("output_size", 1),
                pool_type=config.get("pool_type", "avg"),
                flatten=config.get("flatten", True),
            )

        elif block_type == "attention_pool":
            return AttentionPooling(
                input_dim=config["input_dim"],
                hidden_dim=config.get("hidden_dim", None),
            )

        elif block_type == "global_pool":
            return GlobalPooling(
                pool_type=config.get("pool_type", "avg"),
            )

        # Activation layers
        elif block_type == "relu":
            return nn.ReLU(inplace=True)

        elif block_type == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)

        elif block_type == "gelu":
            return nn.GELU()

        # Normalization layers
        elif block_type == "batch_norm":
            return nn.BatchNorm2d(config["num_features"])

        elif block_type == "dropout":
            return nn.Dropout(config.get("p", 0.5))

        # Linear layers
        elif block_type == "linear":
            return nn.Linear(
                config["in_features"],
                config["out_features"],
                bias=config.get("bias", True),
            )

        # Flatten layer
        elif block_type == "flatten":
            return nn.Flatten(start_dim=config.get("start_dim", 1))

        else:
            raise ValueError(f"Unknown block type: {block_type}")


def create_simple_cnn_config(
    input_channels: int = 1,
    num_classes: int = 10,
    hidden_dims: List[int] = [64, 128, 256],
) -> Dict[str, Any]:
    """Create configuration for a simple CNN.

    Research use: Quick baseline model configuration.

    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions

    Returns:
        Model configuration dictionary
    """
    blocks = []

    # Input conv
    blocks.append({
        "type": "conv",
        "in_channels": input_channels,
        "out_channels": hidden_dims[0],
        "kernel_size": 3,
    })

    # Hidden convs
    for i in range(len(hidden_dims) - 1):
        blocks.append({
            "type": "conv",
            "in_channels": hidden_dims[i],
            "out_channels": hidden_dims[i + 1],
            "kernel_size": 3,
        })

    # Global pooling
    blocks.append({
        "type": "pool",
        "output_size": 1,
        "pool_type": "avg",
        "flatten": True,
    })

    return {
        "blocks": blocks,
        "classifier": {
            "in_features": hidden_dims[-1],
            "num_classes": num_classes,
        },
    }
