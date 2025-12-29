"""Attention mechanisms for neural networks.

Provides attention components for research.
"""

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """Self-attention mechanism for sequences.

    Research use: Capture long-range dependencies in spectrograms/sequences.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads (must divide embed_dim)
        dropout: Dropout probability

    Example:
        >>> attention = SelfAttention(embed_dim=128, num_heads=4)
        >>> # Input: (batch, seq_len, embed_dim)
        >>> out = attention(x)
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through self-attention.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        out = self.proj(out)
        out = self.dropout(out)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention (PyTorch native implementation wrapper).

    Research use: Standard transformer attention.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        batch_first: If True, input is (batch, seq, feature)

    Example:
        >>> attention = MultiHeadAttention(embed_dim=256, num_heads=8)
        >>> out, attn_weights = attention(x, x, x)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        need_weights: bool = False,
    ):
        """Forward pass through multi-head attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attn_mask: Attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights) if need_weights else output
        """
        return self.attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )


class ChannelAttention(nn.Module):
    """Channel attention mechanism (squeeze-and-excitation style).

    Research use: Recalibrate channel-wise feature responses.

    Args:
        channels: Number of channels
        reduction_ratio: Channel reduction ratio for bottleneck

    Example:
        >>> attention = ChannelAttention(channels=128, reduction_ratio=16)
        >>> # Input: (B, C, H, W)
        >>> out = attention(x)  # Same shape as input
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()

        reduced_channels = max(channels // reduction_ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through channel attention.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Attention-weighted tensor (B, C, H, W)
        """
        B, C, _, _ = x.shape

        # Global average pooling
        avg_out = self.avg_pool(x).view(B, C)
        avg_out = self.fc(avg_out)

        # Global max pooling
        max_out = self.max_pool(x).view(B, C)
        max_out = self.fc(max_out)

        # Combine and apply sigmoid
        out = avg_out + max_out
        attention = self.sigmoid(out).view(B, C, 1, 1)

        # Apply attention
        return x * attention
