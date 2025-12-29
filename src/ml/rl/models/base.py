"""Base RL agent interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseRLAgent(ABC):
    """Abstract base class for RL agents."""

    @abstractmethod
    def select_action(self, observation: Any, deterministic: bool = False) -> Any:
        """Select action given observation.

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy

        Returns:
            Action to take
        """
        pass

    @abstractmethod
    def update(self, batch: dict) -> dict[str, float]:
        """Update agent from a batch of experiences.

        Args:
            batch: Batch of (state, action, reward, next_state, done)

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent to disk.

        Args:
            path: Path to save agent checkpoint
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent from disk.

        Args:
            path: Path to load agent checkpoint from
        """
        pass
