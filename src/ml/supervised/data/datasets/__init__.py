"""Dataset classes for supervised learning.

Provides PyTorch Dataset classes for audio data with flexible preprocessing.
"""

from src.ml.supervised.data.datasets.base import BaseAudioDataset
from src.ml.supervised.data.datasets.audio_dataset import (
    AudioClassificationDataset,
    FolderAudioDataset,
    InMemoryAudioDataset,
)

__all__ = [
    "BaseAudioDataset",
    "AudioClassificationDataset",
    "FolderAudioDataset",
    "InMemoryAudioDataset",
]
