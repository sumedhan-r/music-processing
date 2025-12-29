"""Experience replay buffer for RL."""

import random
from collections import deque
from typing import Any, List, Tuple


class ReplayBuffer:
    """Simple experience replay buffer.

    Framework-agnostic implementation that stores experiences
    and provides sampling functionality.
    """

    def __init__(self, capacity: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
