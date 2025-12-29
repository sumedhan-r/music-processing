"""Normalization transforms for preprocessing.

Different normalization strategies for neural network inputs:
- StandardNormalization: Zero mean, unit variance (z-score)
- MinMaxNormalization: Scale to [0, 1] or custom range
- RobustNormalization: Use median and IQR (robust to outliers)
- LogNormalization: Log-scale transformation (for spectrograms)
"""

from src.ml.supervised.data.transforms.normalization.standard import (
    StandardNormalization,
    MinMaxNormalization,
    RobustNormalization,
)
from src.ml.supervised.data.transforms.normalization.audio_specific import (
    LogNormalization,
    DBScaling,
)

__all__ = [
    "StandardNormalization",
    "MinMaxNormalization",
    "RobustNormalization",
    "LogNormalization",
    "DBScaling",
]
