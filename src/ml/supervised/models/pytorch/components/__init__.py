"""Reusable PyTorch model components for building architectures.

Components for composing custom models:
- ConvBlock: Convolutional layer + BatchNorm + Activation
- ResBlock: Residual block for skip connections
- SelfAttention: Self-attention mechanism
- PoolingLayer: Different pooling strategies
"""

from src.ml.supervised.models.pytorch.components.conv_blocks import (
    ConvBlock,
    ResBlock,
    DepthwiseSeparableConv,
)
from src.ml.supervised.models.pytorch.components.attention import (
    SelfAttention,
    MultiHeadAttention,
)
from src.ml.supervised.models.pytorch.components.pooling import (
    AdaptivePooling,
    AttentionPooling,
)

__all__ = [
    "ConvBlock",
    "ResBlock",
    "DepthwiseSeparableConv",
    "SelfAttention",
    "MultiHeadAttention",
    "AdaptivePooling",
    "AttentionPooling",
]
