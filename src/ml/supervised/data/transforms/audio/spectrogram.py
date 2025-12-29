"""Spectrogram transforms for audio preprocessing.

Converts raw audio waveforms to time-frequency representations.
"""

from typing import Optional
import numpy as np

from src.ml.supervised.data.transforms.base import BaseTransform


class AudioToSpectrogram(BaseTransform):
    """Convert audio to spectrogram using STFT.

    Research-friendly: Experiment with different STFT parameters.

    Args:
        n_fft: FFT window size (power of 2, e.g., 512, 1024, 2048)
        hop_length: Number of samples between successive frames
        win_length: Window length (defaults to n_fft)
        window: Window function ('hann', 'hamming', 'blackman')
        center: Whether to center the window
        power: Exponent for magnitude spectrogram (1.0=energy, 2.0=power)

    Example:
        >>> transform = AudioToSpectrogram(n_fft=2048, hop_length=512)
        >>> spectrogram = transform(audio_waveform)
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        power: float = 2.0,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.power = power

    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to spectrogram.

        Args:
            audio: Audio waveform (1D numpy array)

        Returns:
            Spectrogram (2D numpy array: freq x time)

        Note:
            Requires librosa. Install with: pip install librosa
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for spectrogram transforms. "
                "Install with: pip install librosa"
            )

        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
        )

        # Convert to magnitude/power spectrogram
        spectrogram = np.abs(stft) ** self.power

        return spectrogram


class AudioToMelSpectrogram(BaseTransform):
    """Convert audio to mel-scale spectrogram.

    Mel scale approximates human auditory perception.
    Great for music and speech tasks.

    Args:
        sample_rate: Audio sample rate in Hz
        n_fft: FFT window size
        hop_length: Number of samples between frames
        n_mels: Number of mel bands
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz, None = sample_rate/2)
        power: Exponent for magnitude spectrogram

    Example:
        >>> transform = AudioToMelSpectrogram(
        ...     sample_rate=22050,
        ...     n_mels=128,
        ...     fmin=0,
        ...     fmax=8000
        ... )
        >>> mel_spec = transform(audio_waveform)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2.0
        self.power = power

    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mel-spectrogram.

        Args:
            audio: Audio waveform (1D numpy array)

        Returns:
            Mel-spectrogram (2D numpy array: mel_bands x time)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for mel-spectrogram. "
                "Install with: pip install librosa"
            )

        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power,
        )

        return mel_spec
