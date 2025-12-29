"""PyTorch base model implementation."""

import torch
import torch.nn as nn
from typing import Any

from src.ml.supervised.models.base import BaseModel


class PyTorchBaseModel(BaseModel, nn.Module):
    """Base class for PyTorch models.

    Combines the framework-agnostic BaseModel interface with PyTorch's nn.Module.
    """

    def __init__(self):
        """Initialize PyTorch base model."""
        nn.Module.__init__(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_config(self) -> dict[str, Any]:
        """Get model configuration.

        Returns:
            Dictionary containing model configuration
        """
        return {
            "model_class": self.__class__.__name__,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def load_weights(self, path: str) -> None:
        """Load model weights from file.

        Args:
            path: Path to PyTorch checkpoint file (.pt or .pth)
        """
        try:
            state_dict = torch.load(path, map_location="cpu")
            self.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {path}: {e}") from e

    def save_weights(self, path: str) -> None:
        """Save model weights to file.

        Args:
            path: Path to save PyTorch checkpoint file
        """
        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            raise RuntimeError(f"Failed to save weights to {path}: {e}") from e

    def get_device(self) -> torch.device:
        """Get the device the model is on.

        Returns:
            Device (cpu, cuda, mps)
        """
        return next(self.parameters()).device

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with total and trainable parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
