"""Pooling layers for neural networks.

Provides different pooling strategies for research.
"""

from typing import Literal
import torch
import torch.nn as nn


class AdaptivePooling(nn.Module):
    """Adaptive pooling that works with any input size.

    Research use: Flexible pooling for variable-size inputs.

    Args:
        output_size: Target output size (int or tuple)
        pool_type: Pooling type ('avg', 'max', 'avg+max')
        flatten: Whether to flatten output

    Example:
        >>> pool = AdaptivePooling(output_size=1, pool_type='avg', flatten=True)
        >>> # Any input size: (B, C, H, W) -> (B, C)
        >>> out = pool(x)
    """

    def __init__(
        self,
        output_size: int | tuple = 1,
        pool_type: Literal["avg", "max", "avg+max"] = "avg",
        flatten: bool = True,
    ):
        super().__init__()

        self.output_size = output_size
        self.pool_type = pool_type
        self.flatten = flatten

        # Create pooling layers
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        elif pool_type == "avg+max":
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
            self.max_pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adaptive pooling.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Pooled tensor
        """
        if self.pool_type == "avg+max":
            avg_out = self.avg_pool(x)
            max_out = self.max_pool(x)
            out = avg_out + max_out
        else:
            out = self.pool(x)

        if self.flatten:
            out = out.flatten(1)

        return out


class AttentionPooling(nn.Module):
    """Attention-based pooling for aggregating features.

    Research use: Learnable weighted pooling (better than average pooling).

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for attention network

    Example:
        >>> pool = AttentionPooling(input_dim=256, hidden_dim=128)
        >>> # Input: (B, C, H, W)
        >>> out = pool(x)  # (B, C)
    """

    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention pooling.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Pooled tensor (B, C)
        """
        B, C, H, W = x.shape

        # Reshape to (B, H*W, C)
        x = x.view(B, C, H * W).transpose(1, 2)

        # Compute attention weights
        attn_weights = self.attention(x)  # (B, H*W, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Apply attention
        out = (x * attn_weights).sum(dim=1)  # (B, C)

        return out


class GlobalPooling(nn.Module):
    """Global pooling aggregations.

    Research use: Simple baseline pooling strategies.

    Args:
        pool_type: Type of global pooling
            'avg': Global average pooling
            'max': Global max pooling
            'avg+max': Concatenate both
            'std': Standard deviation pooling

    Example:
        >>> pool = GlobalPooling(pool_type='avg+max')
        >>> # (B, C, H, W) -> (B, 2*C)
        >>> out = pool(x)
    """

    def __init__(self, pool_type: Literal["avg", "max", "avg+max", "std"] = "avg"):
        super().__init__()
        self.pool_type = pool_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through global pooling.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Pooled tensor (B, C) or (B, 2*C)
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)

        if self.pool_type == "avg":
            return x_flat.mean(dim=2)

        elif self.pool_type == "max":
            return x_flat.max(dim=2)[0]

        elif self.pool_type == "avg+max":
            avg = x_flat.mean(dim=2)
            max_val = x_flat.max(dim=2)[0]
            return torch.cat([avg, max_val], dim=1)

        elif self.pool_type == "std":
            return x_flat.std(dim=2)

        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
