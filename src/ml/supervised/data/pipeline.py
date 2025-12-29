"""Complete data pipeline for supervised audio learning.

Provides end-to-end data pipeline: Dataset → Transforms → DataLoader.
"""

from typing import Optional, Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader

from src.ml.supervised.data.datasets import (
    AudioClassificationDataset,
    FolderAudioDataset,
)
from src.ml.supervised.data.transforms import TransformPipeline
from src.ml.supervised.data.loaders import DataLoaderFactory


class DataPipeline:
    """Complete data pipeline for audio research.

    Research use: One-stop solution for data loading with transforms.

    Combines:
    - Dataset loading (CSV or folder structure)
    - Transform pipeline (preprocessing, augmentation, normalization)
    - DataLoader creation (batching, shuffling, workers)

    Example:
        >>> from src.ml.supervised.data.pipeline import DataPipeline
        >>> from src.ml.supervised.data.transforms import TransformPipeline
        >>> from src.ml.supervised.data.transforms.audio import AudioToMelSpectrogram
        >>> from src.ml.supervised.data.transforms.normalization import StandardNormalization
        >>>
        >>> # Define transforms
        >>> transform = TransformPipeline([
        ...     AudioToMelSpectrogram(n_mels=128),
        ...     StandardNormalization(),
        ... ])
        >>>
        >>> # Create pipeline
        >>> pipeline = DataPipeline(
        ...     data_dir="data/audio",
        ...     csv_file="data/train.csv",
        ...     transform=transform,
        ...     batch_size=32,
        ...     num_workers=4
        ... )
        >>>
        >>> # Get loaders
        >>> train_loader = pipeline.get_train_loader()
        >>> val_loader = pipeline.get_val_loader()
        >>>
        >>> # Use in training loop
        >>> for audio, labels in train_loader:
        ...     # audio: (batch_size, channels, freq, time)
        ...     # labels: (batch_size,)
        ...     pass
    """

    def __init__(
        self,
        data_dir: str | Path,
        csv_file: Optional[str | Path] = None,
        transform: Optional[TransformPipeline] = None,
        sample_rate: Optional[int] = None,
        max_duration: Optional[float] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        collate_fn: Optional[Callable] = None,
        dataset_type: str = "csv",
    ):
        """Initialize data pipeline.

        Args:
            data_dir: Root directory containing audio files
            csv_file: Path to CSV file (required if dataset_type='csv')
            transform: Transform pipeline to apply
            sample_rate: Target sample rate for audio loading
            max_duration: Maximum audio duration in seconds
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes
            collate_fn: Custom collate function for batching
            dataset_type: Type of dataset ('csv' or 'folder')

        Raises:
            ValueError: If dataset_type is invalid or csv_file missing for 'csv' type
        """
        self.data_dir = Path(data_dir)
        self.csv_file = Path(csv_file) if csv_file else None
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.dataset_type = dataset_type

        # Validate dataset type
        if dataset_type not in ["csv", "folder"]:
            raise ValueError(f"dataset_type must be 'csv' or 'folder', got {dataset_type}")

        if dataset_type == "csv" and csv_file is None:
            raise ValueError("csv_file is required when dataset_type='csv'")

        # Create datasets
        self.train_dataset = self._create_dataset(split="train")
        self.val_dataset = self._create_dataset(split="val")
        self.test_dataset = self._create_dataset(split="test")

    def _create_dataset(self, split: str):
        """Create dataset for given split.

        Args:
            split: Dataset split ('train', 'val', 'test')

        Returns:
            Dataset instance
        """
        if self.dataset_type == "csv":
            return AudioClassificationDataset(
                data_dir=self.data_dir,
                csv_file=self.csv_file,
                split=split,
                transform=self.transform,
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
            )
        else:  # folder
            return FolderAudioDataset(
                data_dir=self.data_dir,
                split=split,
                transform=self.transform,
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
            )

    def get_train_loader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """Get training DataLoader.

        Args:
            batch_size: Override default batch size
            num_workers: Override default num_workers
            collate_fn: Override default collate_fn

        Returns:
            Training DataLoader
        """
        return DataLoaderFactory.create_train_loader(
            dataset=self.train_dataset,
            batch_size=batch_size or self.batch_size,
            num_workers=num_workers or self.num_workers,
            collate_fn=collate_fn or self.collate_fn,
        )

    def get_val_loader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """Get validation DataLoader.

        Args:
            batch_size: Override default batch size
            num_workers: Override default num_workers
            collate_fn: Override default collate_fn

        Returns:
            Validation DataLoader
        """
        return DataLoaderFactory.create_val_loader(
            dataset=self.val_dataset,
            batch_size=batch_size or self.batch_size,
            num_workers=num_workers or self.num_workers,
            collate_fn=collate_fn or self.collate_fn,
        )

    def get_test_loader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """Get test DataLoader.

        Args:
            batch_size: Override default batch size
            num_workers: Override default num_workers
            collate_fn: Override default collate_fn

        Returns:
            Test DataLoader
        """
        return DataLoaderFactory.create_test_loader(
            dataset=self.test_dataset,
            batch_size=batch_size or self.batch_size,
            num_workers=num_workers or self.num_workers,
            collate_fn=collate_fn or self.collate_fn,
        )

    def summary(self) -> dict:
        """Get pipeline summary.

        Returns:
            Dictionary with pipeline statistics
        """
        return {
            "data_dir": str(self.data_dir),
            "csv_file": str(self.csv_file) if self.csv_file else None,
            "dataset_type": self.dataset_type,
            "sample_rate": self.sample_rate,
            "max_duration": self.max_duration,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "transforms": str(self.transform) if self.transform else "None",
        }


class SimplePipeline:
    """Simplified pipeline for quick experimentation.

    Research use: Minimal boilerplate for rapid prototyping.

    Example:
        >>> from src.ml.supervised.data.pipeline import SimplePipeline
        >>>
        >>> pipeline = SimplePipeline.from_folder(
        ...     data_dir="data/audio",
        ...     batch_size=32
        ... )
        >>>
        >>> train_loader = pipeline.train_loader
        >>> val_loader = pipeline.val_loader
    """

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """Initialize with pre-created loaders."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    @staticmethod
    def from_folder(
        data_dir: str | Path,
        transform: Optional[TransformPipeline] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> "SimplePipeline":
        """Create pipeline from folder structure.

        Args:
            data_dir: Root directory with train/val/test folders
            transform: Transform pipeline
            batch_size: Batch size
            num_workers: Number of workers
            sample_rate: Target sample rate
            collate_fn: Custom collate function

        Returns:
            SimplePipeline instance
        """
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_type="folder",
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=sample_rate,
            collate_fn=collate_fn,
        )

        return SimplePipeline(
            train_loader=pipeline.get_train_loader(),
            val_loader=pipeline.get_val_loader(),
            test_loader=pipeline.get_test_loader(),
        )

    @staticmethod
    def from_csv(
        data_dir: str | Path,
        csv_file: str | Path,
        transform: Optional[TransformPipeline] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> "SimplePipeline":
        """Create pipeline from CSV file.

        Args:
            data_dir: Root directory containing audio files
            csv_file: CSV file with file_path,label columns
            transform: Transform pipeline
            batch_size: Batch size
            num_workers: Number of workers
            sample_rate: Target sample rate
            collate_fn: Custom collate function

        Returns:
            SimplePipeline instance
        """
        pipeline = DataPipeline(
            data_dir=data_dir,
            csv_file=csv_file,
            dataset_type="csv",
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=sample_rate,
            collate_fn=collate_fn,
        )

        return SimplePipeline(
            train_loader=pipeline.get_train_loader(),
            val_loader=pipeline.get_val_loader(),
            test_loader=pipeline.get_test_loader(),
        )
