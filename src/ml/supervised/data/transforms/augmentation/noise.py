"""Noise injection augmentation for robustness.

Adding noise to training data improves model robustness and generalization.
"""

from typing import Literal
import numpy as np

from src.ml.supervised.data.transforms.base import BaseTransform


class AddNoise(BaseTransform):
    """Add random noise to data for robustness.

    Research use: Improves generalization and robustness to noisy inputs.

    Args:
        noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
        noise_level: Noise intensity (scale parameter)
        apply_prob: Probability of applying noise (0.0 to 1.0)
        snr_db: Signal-to-noise ratio in dB (alternative to noise_level)

    Example:
        >>> # Gaussian noise with SNR
        >>> transform = AddNoise(noise_type='gaussian', snr_db=20)
        >>> noisy_data = transform(clean_data)

        >>> # Uniform noise
        >>> transform = AddNoise(noise_type='uniform', noise_level=0.01)
        >>> noisy_data = transform(clean_data)
    """

    def __init__(
        self,
        noise_type: Literal["gaussian", "uniform", "salt_pepper"] = "gaussian",
        noise_level: float = 0.01,
        apply_prob: float = 0.5,
        snr_db: float = None,
    ):
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.apply_prob = apply_prob
        self.snr_db = snr_db

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Add noise to data.

        Args:
            data: Input data (any shape)

        Returns:
            Noisy data
        """
        # Decide whether to apply noise
        if np.random.random() > self.apply_prob:
            return data

        data = data.copy()

        # Calculate noise level from SNR if specified
        if self.snr_db is not None:
            signal_power = np.mean(data ** 2)
            snr_linear = 10 ** (self.snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power)
        else:
            noise_std = self.noise_level

        # Generate noise based on type
        if self.noise_type == "gaussian":
            noise = np.random.normal(0, noise_std, data.shape)
            noisy_data = data + noise

        elif self.noise_type == "uniform":
            noise = np.random.uniform(-noise_std, noise_std, data.shape)
            noisy_data = data + noise

        elif self.noise_type == "salt_pepper":
            noisy_data = data.copy()
            # Salt noise (max value)
            salt_mask = np.random.random(data.shape) < (noise_std / 2)
            noisy_data[salt_mask] = np.max(data)
            # Pepper noise (min value)
            pepper_mask = np.random.random(data.shape) < (noise_std / 2)
            noisy_data[pepper_mask] = np.min(data)

        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        return noisy_data


class AddBackgroundNoise(BaseTransform):
    """Add real background noise from a noise dataset.

    Research use: Train on realistic noise conditions.

    Args:
        noise_dataset: List or array of noise samples
        snr_db_range: Range of SNR in dB (min, max)
        apply_prob: Probability of applying noise

    Example:
        >>> # Load background noise samples
        >>> noise_samples = [load_audio(f) for f in noise_files]
        >>> transform = AddBackgroundNoise(
        ...     noise_dataset=noise_samples,
        ...     snr_db_range=(5, 20)
        ... )
        >>> noisy_audio = transform(clean_audio)
    """

    def __init__(
        self,
        noise_dataset: list,
        snr_db_range: tuple = (5, 20),
        apply_prob: float = 0.5,
    ):
        self.noise_dataset = noise_dataset
        self.snr_db_min, self.snr_db_max = snr_db_range
        self.apply_prob = apply_prob

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Add background noise to audio.

        Args:
            data: Clean audio signal

        Returns:
            Noisy audio signal
        """
        # Decide whether to apply noise
        if np.random.random() > self.apply_prob:
            return data

        # Select random noise sample
        noise = np.random.choice(self.noise_dataset)

        # Match lengths (truncate or tile noise)
        if len(noise) < len(data):
            # Tile noise to match length
            repeats = int(np.ceil(len(data) / len(noise)))
            noise = np.tile(noise, repeats)[:len(data)]
        else:
            # Truncate noise
            start = np.random.randint(0, len(noise) - len(data) + 1)
            noise = noise[start:start + len(data)]

        # Random SNR from range
        snr_db = np.random.uniform(self.snr_db_min, self.snr_db_max)

        # Calculate noise scaling factor
        signal_power = np.mean(data ** 2)
        noise_power = np.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(signal_power / (snr_linear * noise_power))

        # Mix signal with scaled noise
        noisy_data = data + scale * noise

        return noisy_data
