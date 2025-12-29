"""Weights & Biases implementation of experiment logger."""

from pathlib import Path
from typing import Any

from src.ml.shared.logging.base import ExperimentLogger


class WandbLogger(ExperimentLogger):
    """Weights & Biases implementation of experiment tracking.

    Uses the wandb Run object API for all operations to ensure proper
    run isolation and support for concurrent runs.

    Example:
        >>> logger = WandbLogger(project="music-processing", name="experiment-1")
        >>> logger.log_metric("loss", 0.5, step=1)
        >>> logger.log_params({"learning_rate": 0.001})
        >>> logger.log_table("predictions", wandb.Table(columns=["x", "y"]))
        >>> logger.finish()
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize wandb logger.

        Args:
            project: Project name
            name: Optional run name
            config: Optional initial configuration
            tags: Optional list of tags for this run
            notes: Optional notes/description for this run
            **kwargs: Additional arguments passed to wandb.init()
        """
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            ) from e

        self.wandb = wandb
        self.run = wandb.init(
            project=project, name=name, config=config, tags=tags, notes=notes, **kwargs
        )

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single metric value using run.log()."""
        self.run.log({name: value}, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once using run.log()."""
        self.run.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters using run.config.update()."""
        self.run.config.update(params)

    def log_artifact(self, path: str | Path, artifact_type: str | None = None) -> None:
        """Log a file artifact using run.log_artifact().

        Args:
            path: Path to the artifact file
            artifact_type: Type of artifact (e.g., 'model', 'dataset', 'plot')
        """
        path = Path(path)
        artifact = self.wandb.Artifact(
            name=path.stem, type=artifact_type or "file"
        )
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def log_table(self, name: str, data: Any) -> None:
        """Log a wandb.Table for visualization using run.log().

        Args:
            name: Table name
            data: wandb.Table object or data convertible to table
        """
        if not isinstance(data, self.wandb.Table):
            # If data is not already a wandb.Table, try to convert it
            # Assumes data is a list of lists or pandas DataFrame
            try:
                data = self.wandb.Table(data=data)
            except Exception as e:
                raise ValueError(
                    f"Cannot convert data to wandb.Table: {e}. "
                    "Please provide a wandb.Table object."
                ) from e

        self.run.log({name: data})

    def save_file(self, path: str | Path) -> None:
        """Save a file to the run using run.save().

        Args:
            path: Path to the file to save
        """
        self.run.save(str(path))

    def finish(self) -> None:
        """Finish the wandb run using run.finish()."""
        self.run.finish()
