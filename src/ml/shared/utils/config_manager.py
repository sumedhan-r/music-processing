"""Repository for configuration management."""

import json
import yaml
from pathlib import Path
from typing import Any

from src.ml.shared.config.schemas import ExperimentConfig


class ConfigRepository:
    """Manages experiment configurations.

    Handles loading, saving, and validating experiment configs
    in a centralized way.
    """

    @staticmethod
    def load_from_yaml(config_path: str | Path) -> ExperimentConfig:
        """Load experiment config from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Validated experiment configuration
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig(**config_dict)

    @staticmethod
    def load_from_json(config_path: str | Path) -> ExperimentConfig:
        """Load experiment config from JSON file.

        Args:
            config_path: Path to JSON config file

        Returns:
            Validated experiment configuration
        """
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return ExperimentConfig(**config_dict)

    @staticmethod
    def save_to_yaml(config: ExperimentConfig, output_path: str | Path) -> None:
        """Save experiment config to YAML file.

        Args:
            config: Experiment configuration
            output_path: Path to save YAML file
        """
        with open(output_path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False)

    @staticmethod
    def save_to_json(config: ExperimentConfig, output_path: str | Path) -> None:
        """Save experiment config to JSON file.

        Args:
            config: Experiment configuration
            output_path: Path to save JSON file
        """
        with open(output_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

    @staticmethod
    def merge_configs(base_config: dict, override_config: dict) -> dict:
        """Merge two configuration dictionaries.

        Args:
            base_config: Base configuration
            override_config: Configuration to override base with

        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        for key, value in override_config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = ConfigRepository.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
