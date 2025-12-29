"""Abstract interface for experiment tracking and logging."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ExperimentLogger(ABC):
    """Abstract base class for experiment tracking tools.

    This interface allows switching between different experiment tracking tools
    (wandb, mlflow, tensorboard) without changing training code.
    """

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single metric value.

        Args:
            name: Metric name (e.g., 'loss', 'accuracy')
            value: Metric value
            step: Optional step/epoch number
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number
        """
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters/hyperparameters.

        Args:
            params: Dictionary of parameter names to values
        """
        pass

    @abstractmethod
    def log_artifact(self, path: str | Path, artifact_type: str | None = None) -> None:
        """Log a file artifact (model, plot, data, etc.).

        Args:
            path: Path to the artifact file
            artifact_type: Optional type label (e.g., 'model', 'plot')
        """
        pass

    @abstractmethod
    def log_table(self, name: str, data: Any) -> None:
        """Log a table for visualization.

        Args:
            name: Table name
            data: Table data (format varies by implementation)
        """
        pass

    @abstractmethod
    def save_file(self, path: str | Path) -> None:
        """Save a file to the experiment run.

        Args:
            path: Path to the file to save
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish the experiment run and cleanup resources."""
        pass
