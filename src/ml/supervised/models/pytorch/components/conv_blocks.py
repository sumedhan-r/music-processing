"""Convolutional building blocks for neural networks.

Provides reusable convolutional components for research.
"""

from typing import Literal, Optional
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Standard convolutional block with normalization and activation.

    Research use: Build custom CNNs by stacking ConvBlocks.

    Architecture: Conv2d -> BatchNorm2d -> Activation -> Optional Dropout

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding (None = 'same' padding)
        activation: Activation function ('relu', 'leaky_relu', 'gelu', 'silu')
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout probability (0 = no dropout)

    Example:
        >>> block = ConvBlock(1, 64, kernel_size=3, activation='relu')
        >>> out = block(x)  # (B, 1, H, W) -> (B, 64, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        activation: Literal["relu", "leaky_relu", "gelu", "silu"] = "relu",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # Auto-calculate padding for 'same' padding
        if padding is None:
            padding = kernel_size // 2

        # Conv layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm,  # No bias if using batch norm
        )

        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    """Residual block with skip connection.

    Research use: Build ResNet-style architectures.

    Architecture:
        input -> Conv -> BN -> Act -> Conv -> BN -> (+input) -> Act

    Args:
        channels: Number of channels (input and output)
        kernel_size: Convolution kernel size
        activation: Activation function
        use_batch_norm: Whether to use batch normalization

    Example:
        >>> block = ResBlock(64, kernel_size=3)
        >>> out = block(x)  # (B, 64, H, W) -> (B, 64, H, W)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        activation: Literal["relu", "leaky_relu", "gelu", "silu"] = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()

        padding = kernel_size // 2

        # First conv block
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=not use_batch_norm)
        self.bn1 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()

        # Second conv block
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out += identity
        out = self.activation(out)

        return out


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (efficient alternative to standard conv).

    Research use: Reduce parameters and computation (MobileNet-style).

    Architecture: Depthwise Conv -> Pointwise Conv (1x1)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Depthwise convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        activation: Activation function
        use_batch_norm: Whether to use batch normalization

    Example:
        >>> # ~9x fewer parameters than standard conv
        >>> block = DepthwiseSeparableConv(64, 128, kernel_size=3)
        >>> out = block(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        activation: Literal["relu", "leaky_relu", "gelu", "silu"] = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        # Depthwise convolution (groups = in_channels)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=not use_batch_norm,
        )
        self.bn1 = nn.BatchNorm2d(in_channels) if use_batch_norm else nn.Identity()

        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=not use_batch_norm,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through depthwise separable conv."""
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)

        return x
