"""Framework-agnostic base model interface for supervised learning.

This module defines the abstract base class for supervised learning models,
providing a common interface for PyTorch, TensorFlow, and other frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    """Abstract base class for supervised learning models.

    This class defines a framework-agnostic interface that all supervised
    learning models should implement, regardless of the underlying framework
    (PyTorch, TensorFlow, etc.).
    """

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through the model.

        Args:
            x: Input data (format depends on framework)

        Returns:
            Model output

        Note:
            - PyTorch: Called via __call__() during forward pass
            - TensorFlow: Maps to call() method
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as a dictionary.

        Returns:
            Dictionary containing model configuration and metadata.
            Typically includes: model_class, num_parameters, trainable_parameters
        """
        pass

    @abstractmethod
    def save_weights(self, filepath: str) -> None:
        """Save model weights to a file.

        Args:
            filepath: Path where weights should be saved
        """
        pass

    @abstractmethod
    def load_weights(self, filepath: str) -> None:
        """Load model weights from a file.

        Args:
            filepath: Path to load weights from
        """
        pass

    def summary(self) -> str:
        """Get a string summary of the model architecture.

        Returns:
            String describing the model architecture

        Note:
            Default implementation returns basic config info.
            Subclasses can override for framework-specific summaries.
        """
        config = self.get_config()
        return f"{self.__class__.__name__}\n" + "\n".join(
            f"  {k}: {v}" for k, v in config.items()
        )
