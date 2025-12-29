"""Base dataset classes for audio data.

Provides abstract base class for building custom audio datasets.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, List
import numpy as np


class BaseAudioDataset(ABC):
    """Abstract base class for audio datasets.

    Research use: Extend this for custom dataset loading strategies.

    Args:
        data_dir: Directory containing audio files
        split: Dataset split ('train', 'val', 'test')
        transform: Transform pipeline to apply to audio
        sample_rate: Target sample rate (None = keep original)
        max_duration: Maximum audio duration in seconds (None = no limit)

    Example:
        >>> class MyDataset(BaseAudioDataset):
        ...     def _load_samples(self):
        ...         # Your custom loading logic
        ...         return file_paths, labels
        ...
        ...     def _load_audio(self, idx):
        ...         # Your custom audio loading
        ...         return audio, label
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Optional[Any] = None,
        sample_rate: Optional[int] = None,
        max_duration: Optional[float] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        # Load dataset samples
        self.samples = self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in {self.data_dir} for split '{split}'"
            )

    @abstractmethod
    def _load_samples(self) -> List[Tuple[Path, Any]]:
        """Load dataset samples (file paths and labels).

        Returns:
            List of (file_path, label) tuples

        Note:
            Implement this in your subclass to define how to find
            audio files and their labels.
        """
        pass

    @abstractmethod
    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load audio file and return waveform.

        Args:
            file_path: Path to audio file

        Returns:
            Audio waveform as numpy array

        Note:
            Implement this in your subclass to handle audio loading
            (librosa, torchaudio, soundfile, etc.)
        """
        pass

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (audio, label) after transforms
        """
        file_path, label = self.samples[idx]

        # Load audio
        audio = self._load_audio(file_path)

        # Apply transforms if provided
        if self.transform is not None:
            audio = self.transform(audio)

        return audio, label

    def get_class_counts(self) -> dict:
        """Get count of samples per class.

        Returns:
            Dictionary mapping class to count
        """
        from collections import Counter

        labels = [label for _, label in self.samples]
        return dict(Counter(labels))

    def get_sample_paths(self) -> List[Path]:
        """Get all sample file paths.

        Returns:
            List of file paths
        """
        return [path for path, _ in self.samples]
