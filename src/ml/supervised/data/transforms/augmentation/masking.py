"""Masking augmentation for spectrograms (SpecAugment).

Based on: "SpecAugment: A Simple Data Augmentation Method for ASR"
https://arxiv.org/abs/1904.08779
"""

from typing import Optional
import numpy as np

from src.ml.supervised.data.transforms.base import BaseTransform


class TimeMask(BaseTransform):
    """Mask contiguous time steps in spectrogram.

    Research use: Improves temporal robustness.

    Args:
        max_mask_length: Maximum number of time steps to mask
        num_masks: Number of masks to apply
        mask_value: Value to use for masking (0 or np.nan)

    Example:
        >>> transform = TimeMask(max_mask_length=10, num_masks=2)
        >>> masked_spec = transform(spectrogram)
    """

    def __init__(
        self,
        max_mask_length: int = 10,
        num_masks: int = 1,
        mask_value: float = 0.0,
    ):
        self.max_mask_length = max_mask_length
        self.num_masks = num_masks
        self.mask_value = mask_value

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply time masking to spectrogram.

        Args:
            data: Spectrogram (freq x time)

        Returns:
            Masked spectrogram
        """
        data = data.copy()
        _, time_steps = data.shape

        for _ in range(self.num_masks):
            # Random mask length
            mask_length = np.random.randint(1, self.max_mask_length + 1)

            # Random starting position
            if time_steps > mask_length:
                start = np.random.randint(0, time_steps - mask_length)
                # Apply mask
                data[:, start:start + mask_length] = self.mask_value

        return data


class FrequencyMask(BaseTransform):
    """Mask contiguous frequency bins in spectrogram.

    Research use: Improves frequency robustness.

    Args:
        max_mask_length: Maximum number of frequency bins to mask
        num_masks: Number of masks to apply
        mask_value: Value to use for masking

    Example:
        >>> transform = FrequencyMask(max_mask_length=8, num_masks=2)
        >>> masked_spec = transform(spectrogram)
    """

    def __init__(
        self,
        max_mask_length: int = 8,
        num_masks: int = 1,
        mask_value: float = 0.0,
    ):
        self.max_mask_length = max_mask_length
        self.num_masks = num_masks
        self.mask_value = mask_value

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply frequency masking to spectrogram.

        Args:
            data: Spectrogram (freq x time)

        Returns:
            Masked spectrogram
        """
        data = data.copy()
        freq_bins, _ = data.shape

        for _ in range(self.num_masks):
            # Random mask length
            mask_length = np.random.randint(1, self.max_mask_length + 1)

            # Random starting position
            if freq_bins > mask_length:
                start = np.random.randint(0, freq_bins - mask_length)
                # Apply mask
                data[start:start + mask_length, :] = self.mask_value

        return data


class SpecAugment(BaseTransform):
    """Combined time and frequency masking (SpecAugment).

    Applies both time and frequency masking as described in the
    SpecAugment paper. Great for speech and audio tasks.

    Args:
        time_mask_max: Maximum time mask length
        freq_mask_max: Maximum frequency mask length
        num_time_masks: Number of time masks
        num_freq_masks: Number of frequency masks
        mask_value: Value to use for masking

    Example:
        >>> transform = SpecAugment(
        ...     time_mask_max=10,
        ...     freq_mask_max=8,
        ...     num_time_masks=2,
        ...     num_freq_masks=2
        ... )
        >>> augmented = transform(spectrogram)
    """

    def __init__(
        self,
        time_mask_max: int = 10,
        freq_mask_max: int = 8,
        num_time_masks: int = 1,
        num_freq_masks: int = 1,
        mask_value: float = 0.0,
    ):
        self.time_masker = TimeMask(
            max_mask_length=time_mask_max,
            num_masks=num_time_masks,
            mask_value=mask_value,
        )
        self.freq_masker = FrequencyMask(
            max_mask_length=freq_mask_max,
            num_masks=num_freq_masks,
            mask_value=mask_value,
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply SpecAugment (time + frequency masking).

        Args:
            data: Spectrogram (freq x time)

        Returns:
            Augmented spectrogram
        """
        # Apply frequency masking first
        data = self.freq_masker.apply(data)
        # Then time masking
        data = self.time_masker.apply(data)
        return data
