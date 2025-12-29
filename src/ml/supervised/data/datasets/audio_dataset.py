"""Concrete audio dataset implementations.

Provides ready-to-use dataset classes for common audio tasks.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict
import numpy as np

from src.ml.supervised.data.datasets.base import BaseAudioDataset


class AudioClassificationDataset(BaseAudioDataset):
    """Audio classification dataset from CSV file.

    Research use: Standard format for audio classification experiments.

    CSV format:
        file_path,label
        audio/train/0.wav,0
        audio/train/1.wav,1

    Args:
        data_dir: Root directory containing audio files
        csv_file: Path to CSV file with file paths and labels
        split: Dataset split ('train', 'val', 'test')
        transform: Transform pipeline
        sample_rate: Target sample rate
        label_map: Optional mapping from string labels to integers

    Example:
        >>> dataset = AudioClassificationDataset(
        ...     data_dir="data/audio",
        ...     csv_file="data/train.csv",
        ...     transform=transform_pipeline
        ... )
        >>> audio, label = dataset[0]
    """

    def __init__(
        self,
        data_dir: str | Path,
        csv_file: str | Path,
        split: str = "train",
        transform: Optional[Any] = None,
        sample_rate: Optional[int] = None,
        max_duration: Optional[float] = None,
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.csv_file = Path(csv_file)
        self.label_map = label_map

        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            sample_rate=sample_rate,
            max_duration=max_duration,
        )

    def _load_samples(self) -> List[Tuple[Path, Any]]:
        """Load samples from CSV file."""
        import csv

        samples = []

        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = self.data_dir / row["file_path"]
                label = row["label"]

                # Convert label if map provided
                if self.label_map is not None:
                    label = self.label_map.get(label, label)
                else:
                    # Try to convert to int
                    try:
                        label = int(label)
                    except ValueError:
                        pass

                if file_path.exists():
                    samples.append((file_path, label))

        return samples

    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load audio using librosa."""
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required. Install with: pip install librosa")

        # Load audio
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

        # Limit duration if specified
        if self.max_duration is not None:
            max_samples = int(self.max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

        return audio


class FolderAudioDataset(BaseAudioDataset):
    """Audio classification dataset organized in folders.

    Research use: Quick setup when data is organized as folders per class.

    Folder structure:
        data_dir/
        ├── train/
        │   ├── class_0/
        │   │   ├── audio1.wav
        │   │   └── audio2.wav
        │   └── class_1/
        │       ├── audio3.wav
        │       └── audio4.wav
        └── val/
            └── ...

    Args:
        data_dir: Root directory
        split: Dataset split ('train', 'val', 'test')
        transform: Transform pipeline
        audio_extensions: File extensions to include
        sample_rate: Target sample rate

    Example:
        >>> dataset = FolderAudioDataset(
        ...     data_dir="data/audio",
        ...     split="train",
        ...     transform=transform_pipeline
        ... )
        >>> print(f"Classes: {dataset.class_names}")
        >>> audio, label = dataset[0]
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Optional[Any] = None,
        sample_rate: Optional[int] = None,
        max_duration: Optional[float] = None,
        audio_extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    ):
        self.audio_extensions = audio_extensions
        self.class_names = []
        self.class_to_idx = {}

        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            sample_rate=sample_rate,
            max_duration=max_duration,
        )

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load samples from folder structure."""
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # Get class folders
        class_folders = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        if len(class_folders) == 0:
            raise ValueError(f"No class folders found in {split_dir}")

        # Build class mapping
        self.class_names = [folder.name for folder in class_folders]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load samples
        samples = []

        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]

            # Find audio files
            for ext in self.audio_extensions:
                for audio_file in class_folder.glob(f"*{ext}"):
                    samples.append((audio_file, class_idx))

        return samples

    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load audio using librosa."""
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required. Install with: pip install librosa")

        # Load audio
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

        # Limit duration if specified
        if self.max_duration is not None:
            max_samples = int(self.max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

        return audio


class InMemoryAudioDataset(BaseAudioDataset):
    """Dataset that loads all audio into memory.

    Research use: Fast iteration when dataset fits in memory.

    Args:
        data_dir: Root directory
        csv_file: Path to CSV file
        split: Dataset split
        transform: Transform pipeline
        sample_rate: Target sample rate
        preload: Whether to load audio immediately

    Example:
        >>> # Fast dataset for small audio clips
        >>> dataset = InMemoryAudioDataset(
        ...     data_dir="data/audio",
        ...     csv_file="data/train.csv",
        ...     preload=True  # Load all audio at init
        ... )
    """

    def __init__(
        self,
        data_dir: str | Path,
        csv_file: str | Path,
        split: str = "train",
        transform: Optional[Any] = None,
        sample_rate: Optional[int] = None,
        max_duration: Optional[float] = None,
        preload: bool = True,
    ):
        self.csv_file = Path(csv_file)
        self.audio_cache = {}

        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            sample_rate=sample_rate,
            max_duration=max_duration,
        )

        if preload:
            self._preload_audio()

    def _load_samples(self) -> List[Tuple[Path, Any]]:
        """Load samples from CSV."""
        import csv

        samples = []

        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = self.data_dir / row["file_path"]
                label = int(row["label"])

                if file_path.exists():
                    samples.append((file_path, label))

        return samples

    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load audio (from cache if preloaded)."""
        # Check cache first
        if file_path in self.audio_cache:
            return self.audio_cache[file_path].copy()

        # Load from disk
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required. Install with: pip install librosa")

        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

        if self.max_duration is not None:
            max_samples = int(self.max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

        return audio

    def _preload_audio(self):
        """Preload all audio into memory."""
        print(f"Preloading {len(self.samples)} audio files...")

        for i, (file_path, _) in enumerate(self.samples):
            if i % 100 == 0:
                print(f"  Loaded {i}/{len(self.samples)}")

            self.audio_cache[file_path] = self._load_audio(file_path)

        print(f"✓ Preloaded {len(self.audio_cache)} audio files")
