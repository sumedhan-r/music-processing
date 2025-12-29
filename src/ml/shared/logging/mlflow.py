"""MLflow implementation of experiment logger."""

from pathlib import Path
from typing import Any

from src.ml.shared.logging.base import ExperimentLogger


class MLflowLogger(ExperimentLogger):
    """MLflow implementation of experiment tracking.

    Example:
        >>> logger = MLflowLogger(experiment_name="music-processing", run_name="exp-1")
        >>> logger.log_metric("loss", 0.5, step=1)
        >>> logger.log_params({"learning_rate": 0.001})
        >>> logger.finish()
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MLflow logger.

        Args:
            experiment_name: Experiment name
            run_name: Optional run name
            tracking_uri: Optional tracking server URI
            tags: Optional tags for this run
            **kwargs: Additional arguments
        """
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "mlflow is not installed. Install it with: pip install mlflow"
            ) from e

        self.mlflow = mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name, tags=tags)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single metric value."""
        self.mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        self.mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters."""
        self.mlflow.log_params(params)

    def log_artifact(self, path: str | Path, artifact_type: str | None = None) -> None:
        """Log a file artifact."""
        self.mlflow.log_artifact(str(path))

    def log_table(self, name: str, data: Any) -> None:
        """Log a table by converting to JSON and logging as artifact.

        Args:
            name: Table name
            data: Table data (pandas DataFrame, dict, or list)
        """
        import json
        import tempfile

        # Try to convert data to JSON-serializable format
        if hasattr(data, "to_dict"):  # pandas DataFrame
            table_data = data.to_dict(orient="records")
        elif isinstance(data, (list, dict)):
            table_data = data
        else:
            raise ValueError(f"Unsupported table data type: {type(data)}")

        # Save as JSON artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(table_data, f, indent=2)
            temp_path = f.name

        self.mlflow.log_artifact(temp_path, artifact_path=f"tables/{name}.json")

    def save_file(self, path: str | Path) -> None:
        """Save a file as an artifact."""
        self.mlflow.log_artifact(str(path))

    def finish(self) -> None:
        """End the MLflow run."""
        self.mlflow.end_run()
