"""OpenAI Gym/Gymnasium environment wrapper."""

from typing import Any, Tuple

from src.ml.shared.environment.base import RLEnvironment


class GymEnvironment(RLEnvironment):
    """Wrapper for OpenAI Gym/Gymnasium environments.

    Provides a consistent interface for both Gym and Gymnasium
    environments by wrapping their API.
    """

    def __init__(self, env: Any):
        """Initialize Gym environment wrapper.

        Args:
            env: Gym or Gymnasium environment instance
        """
        self.env = env
        self._is_gymnasium = self._check_gymnasium()

    def _check_gymnasium(self) -> bool:
        """Check if environment is Gymnasium (new API).

        Returns:
            True if Gymnasium, False if classic Gym
        """
        try:
            # Gymnasium returns (obs, info) from reset
            # Classic Gym returns only obs
            import inspect
            reset_sig = inspect.signature(self.env.reset)
            return "return_info" not in reset_sig.parameters
        except Exception:
            return False

    def reset(self) -> Any:
        """Reset environment to initial state.

        Returns:
            Initial observation
        """
        result = self.env.reset()
        if self._is_gymnasium:
            # Gymnasium returns (obs, info)
            obs, info = result
            return obs
        else:
            # Classic Gym returns obs
            return result

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        result = self.env.step(action)

        if self._is_gymnasium:
            # Gymnasium returns (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated, truncated, info
        else:
            # Classic Gym returns (obs, reward, done, info)
            obs, reward, done, info = result
            # In classic Gym, there's no distinction between terminated and truncated
            return obs, reward, done, False, info

    def get_action_space(self) -> Any:
        """Get the action space definition.

        Returns:
            Action space object (gym.Space)
        """
        return self.env.action_space

    def get_observation_space(self) -> Any:
        """Get the observation space definition.

        Returns:
            Observation space object (gym.Space)
        """
        return self.env.observation_space

    def close(self) -> None:
        """Clean up environment resources."""
        self.env.close()

    def render(self) -> Any:
        """Render the environment.

        Returns:
            Rendered output (depends on render_mode)
        """
        return self.env.render()

    @property
    def unwrapped(self) -> Any:
        """Get the underlying Gym/Gymnasium environment.

        Returns:
            Unwrapped environment instance
        """
        return self.env.unwrapped
