"""Audio-specific normalization transforms."""

from typing import Optional
import numpy as np

from src.ml.supervised.data.transforms.base import BaseTransform


class LogNormalization(BaseTransform):
    """Apply log transformation to compress dynamic range.

    Research use: Common for spectrograms to handle large dynamic range.

    Args:
        offset: Small constant added before log to avoid log(0)
        scale: Scale factor (e.g., 10 for dB scale)
        clip_min: Minimum value after log (for numerical stability)

    Example:
        >>> # Standard log normalization
        >>> transform = LogNormalization()
        >>> log_spec = transform(spectrogram)

        >>> # Decibel scale
        >>> transform = LogNormalization(scale=10)
        >>> db_spec = transform(spectrogram)
    """

    def __init__(
        self,
        offset: float = 1e-10,
        scale: float = 1.0,
        clip_min: Optional[float] = None,
    ):
        self.offset = offset
        self.scale = scale
        self.clip_min = clip_min

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply log transformation.

        Args:
            data: Input data (e.g., spectrogram)

        Returns:
            Log-transformed data
        """
        # Apply log with offset
        log_data = self.scale * np.log(data + self.offset)

        # Optional clipping
        if self.clip_min is not None:
            log_data = np.maximum(log_data, self.clip_min)

        return log_data


class DBScaling(BaseTransform):
    """Convert power spectrogram to decibel (dB) scale.

    Research use: dB scale is perceptually motivated and commonly used.

    Args:
        ref: Reference value for dB calculation (default 1.0)
        amin: Minimum threshold to avoid log(0)
        top_db: Maximum dB value (clips values above this)

    Example:
        >>> transform = DBScaling(ref=1.0, top_db=80)
        >>> db_spec = transform(power_spectrogram)
    """

    def __init__(
        self,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = 80.0,
    ):
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Convert to dB scale.

        Args:
            data: Power spectrogram

        Returns:
            dB-scale spectrogram
        """
        # Clamp to minimum amplitude
        magnitude = np.maximum(self.amin, data)

        # Convert to dB: 10 * log10(S / ref)
        db_spec = 10.0 * np.log10(magnitude / self.ref)

        # Optional top_db clipping
        if self.top_db is not None:
            db_spec = np.maximum(db_spec, db_spec.max() - self.top_db)

        return db_spec


class PerChannelNormalization(BaseTransform):
    """Normalize each frequency channel independently.

    Research use: Useful when different frequency bands have different scales.

    Args:
        method: Normalization method ('standard', 'minmax', 'robust')
        eps: Small constant to avoid division by zero

    Example:
        >>> # Each frequency bin gets normalized independently
        >>> transform = PerChannelNormalization(method='standard')
        >>> normalized = transform(spectrogram)  # (freq, time)
    """

    def __init__(self, method: str = "standard", eps: float = 1e-8):
        self.method = method
        self.eps = eps

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Normalize per frequency channel.

        Args:
            data: Spectrogram (freq x time)

        Returns:
            Per-channel normalized spectrogram
        """
        # Normalize along time axis (axis=1)
        if self.method == "standard":
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            normalized = (data - mean) / (std + self.eps)

        elif self.method == "minmax":
            data_min = np.min(data, axis=1, keepdims=True)
            data_max = np.max(data, axis=1, keepdims=True)
            normalized = (data - data_min) / (data_max - data_min + self.eps)

        elif self.method == "robust":
            median = np.median(data, axis=1, keepdims=True)
            q1 = np.percentile(data, 25, axis=1, keepdims=True)
            q3 = np.percentile(data, 75, axis=1, keepdims=True)
            iqr = q3 - q1
            normalized = (data - median) / (iqr + self.eps)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return normalized
