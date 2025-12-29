"""Abstract interface for data loading."""

from abc import ABC, abstractmethod
from typing import Any, Iterator


class DataLoader(ABC):
    """Abstract base class for data loading.

    This interface allows switching between different data loading strategies
    without changing training code.
    """

    @abstractmethod
    def load_train_data(self) -> Iterator[Any]:
        """Load training data.

        Returns:
            Iterator over training batches
        """
        pass

    @abstractmethod
    def load_val_data(self) -> Iterator[Any]:
        """Load validation data.

        Returns:
            Iterator over validation batches
        """
        pass

    @abstractmethod
    def load_test_data(self) -> Iterator[Any]:
        """Load test data.

        Returns:
            Iterator over test batches
        """
        pass

    @abstractmethod
    def get_batch_size(self) -> int:
        """Get the batch size.

        Returns:
            Batch size
        """
        pass

    @abstractmethod
    def get_dataset_size(self, split: str) -> int:
        """Get the size of a dataset split.

        Args:
            split: Dataset split ('train', 'val', 'test')

        Returns:
            Number of samples in the split
        """
        pass
