"""TensorBoard implementation of experiment logger."""

from pathlib import Path
from typing import Any

from src.ml.shared.logging.base import ExperimentLogger


class TensorBoardLogger(ExperimentLogger):
    """TensorBoard implementation of experiment tracking.

    Example:
        >>> logger = TensorBoardLogger(log_dir="runs/experiment-1")
        >>> logger.log_metric("loss", 0.5, step=1)
        >>> logger.log_params({"learning_rate": 0.001})
        >>> logger.finish()
    """

    def __init__(self, log_dir: str | Path, **kwargs: Any) -> None:
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            **kwargs: Additional arguments passed to SummaryWriter
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "tensorboard is not installed. Install it with: pip install tensorboard"
            ) from e

        self.writer = SummaryWriter(log_dir=str(log_dir), **kwargs)
        self.log_dir = Path(log_dir)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single metric value."""
        if step is None:
            step = 0
        self.writer.add_scalar(name, value, step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        if step is None:
            step = 0
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters as text."""
        params_str = "\n".join(f"{k}: {v}" for k, v in params.items())
        self.writer.add_text("hyperparameters", params_str)

    def log_artifact(self, path: str | Path, artifact_type: str | None = None) -> None:
        """Log a file artifact by copying to log directory."""
        import shutil

        path = Path(path)
        artifact_dir = self.log_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        dest = artifact_dir / path.name
        shutil.copy(path, dest)

    def log_table(self, name: str, data: Any) -> None:
        """Log a table as markdown text.

        Args:
            name: Table name
            data: Table data (pandas DataFrame, dict, or list)
        """
        # Convert data to markdown format
        if hasattr(data, "to_markdown"):  # pandas DataFrame
            table_md = data.to_markdown()
        elif isinstance(data, list) and len(data) > 0:
            # Simple list of dicts to markdown
            if isinstance(data[0], dict):
                headers = list(data[0].keys())
                rows = [[str(row.get(h, "")) for h in headers] for row in data]
                table_md = "| " + " | ".join(headers) + " |\n"
                table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in rows:
                    table_md += "| " + " | ".join(row) + " |\n"
            else:
                table_md = str(data)
        else:
            table_md = str(data)

        self.writer.add_text(f"table/{name}", table_md)

    def save_file(self, path: str | Path) -> None:
        """Save a file by copying to log directory."""
        import shutil

        path = Path(path)
        dest = self.log_dir / path.name
        shutil.copy(path, dest)

    def finish(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
