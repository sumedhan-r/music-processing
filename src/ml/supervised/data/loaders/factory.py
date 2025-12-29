"""DataLoader factory for creating PyTorch DataLoaders.

Provides convenient DataLoader creation with sensible defaults for audio research.
"""

from typing import Optional, Any, Callable
from torch.utils.data import DataLoader, Dataset


class DataLoaderFactory:
    """Factory for creating PyTorch DataLoaders with research-friendly defaults.

    Research use: Quick DataLoader setup with common configurations.

    Example:
        >>> from src.ml.supervised.data.datasets import AudioClassificationDataset
        >>> from src.ml.supervised.data.loaders import DataLoaderFactory
        >>>
        >>> dataset = AudioClassificationDataset(...)
        >>> loader = DataLoaderFactory.create(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     num_workers=4
        ... )
    """

    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """Create a DataLoader with sensible defaults.

        Args:
            dataset: PyTorch Dataset
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            num_workers: Number of subprocesses for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop last incomplete batch
            collate_fn: Custom collate function for batching
            **kwargs: Additional arguments passed to DataLoader

        Returns:
            Configured PyTorch DataLoader

        Example:
            >>> loader = DataLoaderFactory.create(
            ...     dataset,
            ...     batch_size=64,
            ...     num_workers=8,
            ...     collate_fn=PadCollate(max_length=100000)
            ... )
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs,
        )

    @staticmethod
    def create_train_loader(
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        collate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """Create DataLoader for training.

        Includes shuffling and drops incomplete batches.

        Args:
            dataset: Training dataset
            batch_size: Batch size
            num_workers: Number of workers
            collate_fn: Custom collate function
            **kwargs: Additional DataLoader arguments

        Returns:
            Training DataLoader
        """
        return DataLoaderFactory.create(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            **kwargs,
        )

    @staticmethod
    def create_val_loader(
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        collate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """Create DataLoader for validation.

        No shuffling, keeps all samples.

        Args:
            dataset: Validation dataset
            batch_size: Batch size
            num_workers: Number of workers
            collate_fn: Custom collate function
            **kwargs: Additional DataLoader arguments

        Returns:
            Validation DataLoader
        """
        return DataLoaderFactory.create(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_fn,
            **kwargs,
        )

    @staticmethod
    def create_test_loader(
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        collate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """Create DataLoader for testing.

        No shuffling, keeps all samples, single-sample batches supported.

        Args:
            dataset: Test dataset
            batch_size: Batch size
            num_workers: Number of workers
            collate_fn: Custom collate function
            **kwargs: Additional DataLoader arguments

        Returns:
            Test DataLoader
        """
        return DataLoaderFactory.create(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_fn,
            **kwargs,
        )
