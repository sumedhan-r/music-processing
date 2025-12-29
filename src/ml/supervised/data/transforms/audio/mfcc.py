"""MFCC (Mel-Frequency Cepstral Coefficients) transforms.

MFCCs are commonly used in speech recognition and music analysis.
"""

from typing import Optional
import numpy as np

from src.ml.supervised.data.transforms.base import BaseTransform


class AudioToMFCC(BaseTransform):
    """Convert audio to MFCC features.

    MFCCs capture spectral envelope information and are
    widely used in speech and audio classification tasks.

    Args:
        sample_rate: Audio sample rate in Hz
        n_mfcc: Number of MFCC coefficients to extract
        n_fft: FFT window size
        hop_length: Number of samples between frames
        n_mels: Number of mel bands for mel-spectrogram
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz, None = sample_rate/2)
        lifter: Cepstral filtering coefficient (0 = no filtering)
        dct_type: Type of DCT (Discrete Cosine Transform) to use

    Example:
        >>> transform = AudioToMFCC(
        ...     sample_rate=22050,
        ...     n_mfcc=13,  # 13 coefficients is standard
        ...     n_mels=40
        ... )
        >>> mfcc = transform(audio_waveform)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        lifter: int = 0,
        dct_type: int = 2,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2.0
        self.lifter = lifter
        self.dct_type = dct_type

    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to MFCC features.

        Args:
            audio: Audio waveform (1D numpy array)

        Returns:
            MFCC features (2D numpy array: n_mfcc x time)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for MFCC extraction. "
                "Install with: pip install librosa"
            )

        # Compute MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            lifter=self.lifter,
            dct_type=self.dct_type,
        )

        return mfcc


class AudioToMFCCDelta(BaseTransform):
    """Compute MFCC delta (velocity) and delta-delta (acceleration) features.

    Delta features capture temporal dynamics and improve classification.

    Args:
        sample_rate: Audio sample rate in Hz
        n_mfcc: Number of MFCC coefficients
        include_deltas: Include first derivatives (velocity)
        include_delta_deltas: Include second derivatives (acceleration)
        **mfcc_kwargs: Additional arguments for MFCC extraction

    Example:
        >>> transform = AudioToMFCCDelta(
        ...     sample_rate=22050,
        ...     n_mfcc=13,
        ...     include_deltas=True,
        ...     include_delta_deltas=True
        ... )
        >>> # Returns stacked [MFCC, Delta, Delta-Delta]
        >>> features = transform(audio_waveform)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 13,
        include_deltas: bool = True,
        include_delta_deltas: bool = True,
        **mfcc_kwargs,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.include_deltas = include_deltas
        self.include_delta_deltas = include_delta_deltas
        self.mfcc_kwargs = mfcc_kwargs

        # Create base MFCC transform
        self.mfcc_transform = AudioToMFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            **mfcc_kwargs
        )

    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Compute MFCC with delta features.

        Args:
            audio: Audio waveform (1D numpy array)

        Returns:
            Stacked features (2D numpy array)
            Shape: (n_mfcc * [1 + deltas + delta_deltas]) x time
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for MFCC delta features. "
                "Install with: pip install librosa"
            )

        # Compute base MFCC
        mfcc = self.mfcc_transform.apply(audio)

        features = [mfcc]

        # Compute deltas
        if self.include_deltas:
            delta = librosa.feature.delta(mfcc, order=1)
            features.append(delta)

        # Compute delta-deltas
        if self.include_delta_deltas:
            delta_delta = librosa.feature.delta(mfcc, order=2)
            features.append(delta_delta)

        # Stack all features
        return np.vstack(features)
