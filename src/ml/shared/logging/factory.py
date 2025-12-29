"""Factory for creating experiment loggers."""

from typing import Any

from src.ml.shared.logging.base import ExperimentLogger
from src.ml.shared.config.schemas import LoggerConfig


class LoggerFactory:
    """Factory for creating experiment loggers based on configuration."""

    @staticmethod
    def create(config: LoggerConfig, **kwargs: Any) -> ExperimentLogger:
        """Create an experiment logger based on the provider specified in config.

        Args:
            config: Logger configuration specifying provider and settings
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured ExperimentLogger instance

        Raises:
            ValueError: If provider is not supported
            ImportError: If provider package is not installed
        """
        provider = config.provider.lower()

        if provider == "wandb":
            from src.ml.shared.logging.wandb import WandbLogger

            return WandbLogger(
                project=config.project_name,
                name=config.experiment_name,
                tags=config.tags,
                notes=config.notes,
                **kwargs,
            )
        elif provider == "mlflow":
            from src.ml.shared.logging.mlflow import MLflowLogger

            return MLflowLogger(
                experiment_name=config.project_name,
                run_name=config.experiment_name,
                tags=config.tags,
                **kwargs,
            )
        elif provider == "tensorboard":
            from src.ml.shared.logging.tensorboard import TensorboardLogger

            return TensorboardLogger(
                log_dir=f"runs/{config.project_name}/{config.experiment_name}",
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported logger provider: {provider}. "
                f"Supported providers: wandb, mlflow, tensorboard"
            )

    @staticmethod
    def create_from_name(
        provider: str,
        project: str,
        name: str | None = None,
        **kwargs: Any,
    ) -> ExperimentLogger:
        """Create a logger directly from provider name and basic settings.

        Args:
            provider: Logger provider ('wandb', 'mlflow', 'tensorboard')
            project: Project name
            name: Optional experiment/run name
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured ExperimentLogger instance

        Raises:
            ValueError: If provider is not supported
            ImportError: If provider package is not installed
        """
        # Create minimal config and use main create method
        config = LoggerConfig(
            provider=provider,
            project_name=project,
            experiment_name=name or "experiment",
        )
        return LoggerFactory.create(config, **kwargs)
