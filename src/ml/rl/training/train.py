"""RL training loop."""

from typing import Any

from src.ml.shared.environment.base import RLEnvironment
from src.ml.shared.logging.base import ExperimentLogger
from src.ml.rl.models.base import BaseRLAgent
from src.ml.rl.replay_buffer import ReplayBuffer
from src.ml.shared.config.schemas import RLExperimentConfig


def train_rl_agent(
    config: RLExperimentConfig,
    agent: BaseRLAgent,
    env: RLEnvironment,
    replay_buffer: ReplayBuffer,
    logger: ExperimentLogger,
) -> dict[str, Any]:
    """Main RL training loop.

    Args:
        config: RL experiment configuration
        agent: RL agent to train
        env: Training environment
        replay_buffer: Experience replay buffer
        logger: Experiment logger

    Returns:
        Dictionary containing training results
    """
    # Log configuration
    logger.log_params(config.model_dump())

    episode_rewards = []
    episode = 0
    observation, _ = env.reset()

    for timestep in range(config.training.total_timesteps):
        # Warmup: random actions
        if timestep < config.training.warmup_steps:
            action = env.get_action_space().sample()  # Random action
        else:
            action = agent.select_action(observation, deterministic=False)

        # Environment step
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store experience
        replay_buffer.add(observation, action, reward, next_observation, done)

        # Training step
        if (
            timestep >= config.training.warmup_steps
            and timestep % config.training.train_freq == 0
            and len(replay_buffer) >= config.agent.replay_buffer.batch_size
        ):
            for _ in range(config.training.gradient_steps):
                batch = replay_buffer.sample(config.agent.replay_buffer.batch_size)
                metrics = agent.update(batch)

                # Log training metrics
                if timestep % 100 == 0:
                    logger.log_metrics(metrics, step=timestep)

        # Episode end
        if done:
            episode += 1
            episode_rewards.append(info.get("episode_reward", 0))

            logger.log_metric("episode_reward", episode_rewards[-1], step=timestep)
            logger.log_metric("episode", episode, step=timestep)

            observation, _ = env.reset()
        else:
            observation = next_observation

        # Evaluation
        if timestep % config.training.eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, config.training.eval_episodes)
            logger.log_metric("eval_reward", eval_reward, step=timestep)

        # Save checkpoint
        if timestep % config.training.save_freq == 0:
            agent.save(f"checkpoint_{timestep}.pth")

    return {
        "total_episodes": episode,
        "mean_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
    }


def evaluate_agent(
    agent: BaseRLAgent,
    env: RLEnvironment,
    num_episodes: int,
) -> float:
    """Evaluate agent performance.

    Args:
        agent: Agent to evaluate
        env: Evaluation environment
        num_episodes: Number of episodes to evaluate

    Returns:
        Mean episode reward
    """
    total_rewards = []

    for _ in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    return sum(total_rewards) / len(total_rewards) if total_rewards else 0
