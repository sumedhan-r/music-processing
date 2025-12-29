"""Standard normalization transforms."""

from typing import Optional, Tuple
import numpy as np

from src.ml.supervised.data.transforms.base import BaseTransform


class StandardNormalization(BaseTransform):
    """Standardize data to zero mean and unit variance (z-score normalization).

    Research use: Most common normalization for neural networks.

    Args:
        axis: Axis along which to compute statistics (None = global)
        eps: Small constant to avoid division by zero

    Example:
        >>> # Global normalization
        >>> transform = StandardNormalization()
        >>> normalized = transform(data)

        >>> # Per-feature normalization
        >>> transform = StandardNormalization(axis=0)
        >>> normalized = transform(data)
    """

    def __init__(self, axis: Optional[int] = None, eps: float = 1e-8):
        self.axis = axis
        self.eps = eps

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Standardize data.

        Args:
            data: Input data

        Returns:
            Standardized data with mean=0, std=1
        """
        mean = np.mean(data, axis=self.axis, keepdims=True)
        std = np.std(data, axis=self.axis, keepdims=True)

        normalized = (data - mean) / (std + self.eps)

        return normalized


class MinMaxNormalization(BaseTransform):
    """Scale data to a specified range (default [0, 1]).

    Research use: Good for data with known bounds.

    Args:
        feature_range: Target range (min, max)
        axis: Axis along which to compute statistics (None = global)
        eps: Small constant to avoid division by zero

    Example:
        >>> # Scale to [0, 1]
        >>> transform = MinMaxNormalization()
        >>> normalized = transform(data)

        >>> # Scale to [-1, 1]
        >>> transform = MinMaxNormalization(feature_range=(-1, 1))
        >>> normalized = transform(data)
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        axis: Optional[int] = None,
        eps: float = 1e-8,
    ):
        self.feature_range = feature_range
        self.axis = axis
        self.eps = eps

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Scale data to feature range.

        Args:
            data: Input data

        Returns:
            Scaled data in specified range
        """
        data_min = np.min(data, axis=self.axis, keepdims=True)
        data_max = np.max(data, axis=self.axis, keepdims=True)

        # Scale to [0, 1]
        scaled = (data - data_min) / (data_max - data_min + self.eps)

        # Scale to feature_range
        range_min, range_max = self.feature_range
        normalized = scaled * (range_max - range_min) + range_min

        return normalized


class RobustNormalization(BaseTransform):
    """Normalize using median and IQR (robust to outliers).

    Research use: Better than StandardNormalization when data has outliers.

    Args:
        axis: Axis along which to compute statistics (None = global)
        eps: Small constant to avoid division by zero
        quantile_range: Percentile range for IQR (default 25-75%)

    Example:
        >>> transform = RobustNormalization()
        >>> normalized = transform(data)
    """

    def __init__(
        self,
        axis: Optional[int] = None,
        eps: float = 1e-8,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
    ):
        self.axis = axis
        self.eps = eps
        self.quantile_range = quantile_range

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Normalize using robust statistics.

        Args:
            data: Input data

        Returns:
            Robustly normalized data
        """
        median = np.median(data, axis=self.axis, keepdims=True)

        q_low, q_high = self.quantile_range
        q1 = np.percentile(data, q_low, axis=self.axis, keepdims=True)
        q3 = np.percentile(data, q_high, axis=self.axis, keepdims=True)
        iqr = q3 - q1

        normalized = (data - median) / (iqr + self.eps)

        return normalized
