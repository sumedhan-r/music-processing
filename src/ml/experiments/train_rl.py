"""Example reinforcement learning experiment script.

This script demonstrates how to use the reorganized codebase to train
an RL agent with experiment tracking.
"""

from src.ml.shared.config.schemas import (
    RLExperimentConfig,
    RLEnvironmentConfig,
    RLAgentConfig,
    RLTrainingConfig,
    ReplayBufferConfig,
)
from src.ml.shared.logging.factory import LoggerFactory
from src.ml.shared.logging.base import ExperimentLogger
from src.ml.shared.environment import GymEnvironment
from src.ml.rl.replay_buffer import ReplayBuffer
from src.ml.rl.training.train import train_rl_agent


def create_rl_experiment_config() -> RLExperimentConfig:
    """Create a sample RL experiment configuration.

    Returns:
        Complete RL experiment configuration
    """
    # Environment configuration
    env_config = RLEnvironmentConfig(
        env_id="CartPole-v1",
        env_type="gym",
        num_envs=1,
        max_episode_steps=500,
    )

    # Agent configuration
    agent_config = RLAgentConfig(
        algorithm="dqn",
        gamma=0.99,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=1000,
    )

    # Replay buffer configuration
    replay_buffer_config = ReplayBufferConfig(
        capacity=10000,
        batch_size=64,
        prioritized=False,
    )

    # Training configuration
    training_config = RLTrainingConfig(
        total_timesteps=50000,
        warmup_steps=1000,
        train_freq=4,
        gradient_steps=1,
        eval_freq=5000,
        eval_episodes=10,
        save_freq=10000,
        replay_buffer=replay_buffer_config,
    )

    return RLExperimentConfig(
        experiment_id="rl-cartpole-dqn-exp1",
        environment=env_config,
        agent=agent_config,
        training=training_config,
        seed=42,
    )


def main():
    """Run the RL experiment."""
    # 1. Create experiment configuration
    config = create_rl_experiment_config()

    # 2. Initialize logger (using direct method for RL)
    logger = LoggerFactory.create_from_name(
        provider="tensorboard",  # or "wandb", "mlflow"
        project="music-processing-rl",
        name=config.experiment_id,
    )

    # 3. Log experiment parameters
    logger.log_params(config.model_dump())

    # 4. TODO: Initialize environment
    # import gymnasium as gym
    # env = gym.make(config.environment.env_id)
    # wrapped_env = GymEnvironment(env)

    # 5. TODO: Initialize agent
    # from src.ml.rl.models.base import BaseRLAgent
    # agent = YourRLAgent(config.agent)

    # 6. TODO: Initialize replay buffer
    # replay_buffer = ReplayBuffer(capacity=config.training.replay_buffer.capacity)

    # 7. TODO: Train the agent
    # train_rl_agent(
    #     config=config,
    #     agent=agent,
    #     env=wrapped_env,
    #     replay_buffer=replay_buffer,
    #     logger=logger,
    # )

    # 8. Finish logging
    logger.finish()

    print("✓ RL Experiment completed!")


if __name__ == "__main__":
    main()
