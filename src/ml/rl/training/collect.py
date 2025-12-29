"""Experience collection utilities for RL."""

from typing import Any, Dict, List, Tuple

from src.ml.shared.environment.base import RLEnvironment
from src.ml.rl.models.base import BaseRLAgent
from src.ml.rl.replay_buffer import ReplayBuffer


def collect_experience(
    agent: BaseRLAgent,
    env: RLEnvironment,
    replay_buffer: ReplayBuffer,
    num_steps: int,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """Collect experience by interacting with the environment.

    Args:
        agent: Agent to use for action selection
        env: Environment to interact with
        replay_buffer: Buffer to store experiences
        num_steps: Number of steps to collect
        deterministic: Whether to use deterministic policy

    Returns:
        Dictionary containing collection statistics
    """
    observation, _ = env.reset()
    total_reward = 0
    episodes_completed = 0
    steps_collected = 0

    for _ in range(num_steps):
        action = agent.select_action(observation, deterministic=deterministic)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.add(observation, action, reward, next_observation, done)

        total_reward += reward
        steps_collected += 1

        if done:
            episodes_completed += 1
            observation, _ = env.reset()
        else:
            observation = next_observation

    return {
        "steps_collected": steps_collected,
        "episodes_completed": episodes_completed,
        "total_reward": total_reward,
        "buffer_size": len(replay_buffer),
    }


def collect_rollout(
    agent: BaseRLAgent,
    env: RLEnvironment,
    max_steps: int = 1000,
    deterministic: bool = True,
) -> Tuple[List[Any], List[Any], List[float], Dict[str, Any]]:
    """Collect a complete episode rollout.

    Args:
        agent: Agent to use for action selection
        env: Environment to interact with
        max_steps: Maximum steps per episode
        deterministic: Whether to use deterministic policy

    Returns:
        Tuple of (states, actions, rewards, info)
    """
    states = []
    actions = []
    rewards = []

    observation, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        states.append(observation)
        action = agent.select_action(observation, deterministic=deterministic)
        actions.append(action)

        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        total_reward += reward
        steps += 1
        done = terminated or truncated

    info = {
        "total_reward": total_reward,
        "episode_length": steps,
        "terminated": terminated if done else False,
        "truncated": truncated if done else True,
    }

    return states, actions, rewards, info
