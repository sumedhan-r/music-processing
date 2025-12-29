"""Base transform classes for preprocessing pipelines.

Provides abstract base class and pipeline composition for research workflows.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseTransform(ABC):
    """Abstract base class for all transforms.

    Design for research: Easy to create new transforms by subclassing.

    Example:
        >>> class MyCustomTransform(BaseTransform):
        ...     def __init__(self, param1, param2):
        ...         self.param1 = param1
        ...         self.param2 = param2
        ...
        ...     def apply(self, data):
        ...         # Your preprocessing logic
        ...         return processed_data
    """

    @abstractmethod
    def apply(self, data: Any) -> Any:
        """Apply the transform to data.

        Args:
            data: Input data (numpy array, tensor, audio signal, etc.)

        Returns:
            Transformed data
        """
        pass

    def __call__(self, data: Any) -> Any:
        """Allow using transform as a callable."""
        return self.apply(data)

    def __repr__(self) -> str:
        """String representation for debugging."""
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"


class TransformPipeline:
    """Compose multiple transforms into a sequential pipeline.

    Design for research: Easy to experiment with different preprocessing chains.

    Example:
        >>> pipeline = TransformPipeline([
        ...     AudioToSpectrogram(n_fft=2048),
        ...     FreqMask(max_mask=8),
        ...     StandardNormalization(),
        ... ])
        >>> processed = pipeline(audio_data)
    """

    def __init__(self, transforms: List[BaseTransform]):
        """Initialize pipeline with list of transforms.

        Args:
            transforms: List of transform objects to apply sequentially
        """
        self.transforms = transforms

    def apply(self, data: Any) -> Any:
        """Apply all transforms in sequence.

        Args:
            data: Input data

        Returns:
            Data after applying all transforms
        """
        for transform in self.transforms:
            data = transform.apply(data)
        return data

    def __call__(self, data: Any) -> Any:
        """Allow using pipeline as a callable."""
        return self.apply(data)

    def __repr__(self) -> str:
        """String representation showing all transforms."""
        transform_strs = "\n  ".join(str(t) for t in self.transforms)
        return f"TransformPipeline([\n  {transform_strs}\n])"

    def __len__(self) -> int:
        """Number of transforms in pipeline."""
        return len(self.transforms)

    def __getitem__(self, idx: int) -> BaseTransform:
        """Get transform by index."""
        return self.transforms[idx]

    def add(self, transform: BaseTransform) -> "TransformPipeline":
        """Add a transform to the pipeline.

        Args:
            transform: Transform to add

        Returns:
            Self for method chaining
        """
        self.transforms.append(transform)
        return self

    def insert(self, idx: int, transform: BaseTransform) -> "TransformPipeline":
        """Insert a transform at specific position.

        Args:
            idx: Index to insert at
            transform: Transform to insert

        Returns:
            Self for method chaining
        """
        self.transforms.insert(idx, transform)
        return self

    def remove(self, idx: int) -> "TransformPipeline":
        """Remove a transform by index.

        Args:
            idx: Index of transform to remove

        Returns:
            Self for method chaining
        """
        self.transforms.pop(idx)
        return self
