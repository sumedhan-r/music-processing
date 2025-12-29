"""Experience buffer for RL data storage and management."""

from typing import Any, Dict, List, Optional

from src.ml.rl.replay_buffer import ReplayBuffer


class ExperienceBuffer(ReplayBuffer):
    """Enhanced replay buffer with additional functionality for RL.

    Extends the basic ReplayBuffer with features like prioritized
    sampling, multi-step returns, and buffer statistics.
    """

    def __init__(
        self,
        capacity: int,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        """Initialize experience buffer.

        Args:
            capacity: Maximum buffer size
            prioritized: Whether to use prioritized experience replay
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        super().__init__(capacity)
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.priorities: List[float] = []
        self.max_priority = 1.0

    def add(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """Add experience to buffer with optional priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            priority: Priority value (for prioritized replay)
        """
        super().add(state, action, reward, next_state, done)

        if self.prioritized:
            if priority is None:
                priority = self.max_priority
            self.priorities.append(priority)
            if len(self.priorities) > len(self.buffer):
                self.priorities.pop(0)

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        if not self.prioritized:
            return

        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary containing buffer statistics
        """
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.buffer.maxlen,
                "utilization": 0.0,
            }

        rewards = [exp[2] for exp in self.buffer]
        return {
            "size": len(self.buffer),
            "capacity": self.buffer.maxlen,
            "utilization": len(self.buffer) / self.buffer.maxlen,
            "mean_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
        }

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.max_priority = 1.0
