"""Abstract interface for RL environments."""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class RLEnvironment(ABC):
    """Abstract base class for RL environments.

    Provides a framework-agnostic interface that can wrap
    OpenAI Gym, Gymnasium, or custom environments.
    """

    @abstractmethod
    def reset(self) -> Any:
        """Reset environment to initial state.

        Returns:
            Initial observation
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass

    @abstractmethod
    def get_action_space(self) -> Any:
        """Get the action space definition."""
        pass

    @abstractmethod
    def get_observation_space(self) -> Any:
        """Get the observation space definition."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass
