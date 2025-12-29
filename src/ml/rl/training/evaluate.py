"""RL evaluation utilities."""

from typing import Any, Dict, List

from src.ml.shared.environment.base import RLEnvironment
from src.ml.rl.models.base import BaseRLAgent


def evaluate_policy(
    agent: BaseRLAgent,
    env: RLEnvironment,
    num_episodes: int = 10,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Evaluate agent policy over multiple episodes.

    Args:
        agent: Agent to evaluate
        env: Evaluation environment
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic policy

    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    for _ in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return {
        "mean_reward": sum(episode_rewards) / len(episode_rewards),
        "std_reward": _std(episode_rewards),
        "min_reward": min(episode_rewards),
        "max_reward": max(episode_rewards),
        "mean_length": sum(episode_lengths) / len(episode_lengths),
        "episodes": num_episodes,
    }


def _std(values: List[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5
