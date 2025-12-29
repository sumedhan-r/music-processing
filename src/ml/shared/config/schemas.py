"""Configuration schemas for ML experiments.

Consolidated schemas from:
- Data and features
- Models and inference
- Training and optimization
- Experiment tracking
- RL-specific configurations
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Data and Feature Schemas
# ============================================================================


class AudioFeatures(BaseModel):
    """Audio feature representation.

    Note: For numpy arrays and tensors, validation is minimal.
    Use with arbitrary_types_allowed = True in Config.
    """

    sample_rate: int = Field(gt=0, description="Audio sample rate in Hz")
    duration: float = Field(gt=0, description="Audio duration in seconds")
    num_channels: int = Field(default=1, ge=1, description="Number of audio channels")
    feature_type: str = Field(description="Type of features (e.g., 'spectrogram', 'mfcc')")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(description="Dataset name")
    data_dir: str = Field(description="Path to data directory")
    train_split: float = Field(gt=0, lt=1, default=0.8, description="Training split ratio")
    val_split: float = Field(gt=0, lt=1, default=0.1, description="Validation split ratio")
    test_split: float = Field(gt=0, lt=1, default=0.1, description="Test split ratio")
    shuffle: bool = Field(default=True, description="Shuffle dataset")
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    preprocessing: dict[str, Any] | None = Field(
        default=None, description="Preprocessing configuration"
    )


class BatchData(BaseModel):
    """Batch of training/validation data.

    Generic schema - actual implementation will vary by task.
    """

    batch_size: int = Field(gt=0, description="Number of samples in batch")
    features: Any = Field(description="Batch features (tensors, arrays, etc.)")
    labels: Any = Field(description="Batch labels")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional batch metadata"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


# ============================================================================
# Model Schemas
# ============================================================================


class ModelConfig(BaseModel):
    """Base model configuration."""

    model_name: str = Field(description="Model name/identifier")
    model_type: str = Field(description="Type of model architecture")
    input_shape: tuple[int, ...] = Field(description="Input tensor shape")
    output_shape: tuple[int, ...] = Field(description="Output tensor shape")
    num_classes: int | None = Field(default=None, gt=0, description="Number of classes for classification")
    dropout_rate: float = Field(default=0.0, ge=0, le=1, description="Dropout rate")
    use_batch_norm: bool = Field(default=True, description="Use batch normalization")


class ModelCheckpoint(BaseModel):
    """Model checkpoint metadata."""

    checkpoint_path: str = Field(description="Path to checkpoint file")
    epoch: int = Field(ge=0, description="Training epoch")
    step: int = Field(ge=0, description="Training step")
    metrics: dict[str, float] = Field(description="Metrics at checkpoint time")
    timestamp: str = Field(description="Checkpoint timestamp")
    config: dict[str, Any] = Field(description="Model and training config")


class ModelPrediction(BaseModel):
    """Model prediction output."""

    predictions: Any = Field(description="Model predictions")
    confidence: float | list[float] | None = Field(
        default=None, ge=0, le=1, description="Prediction confidence scores"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional prediction metadata"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class InferenceConfig(BaseModel):
    """Inference configuration."""

    model_path: str = Field(description="Path to trained model")
    batch_size: int = Field(default=32, gt=0, description="Inference batch size")
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device for inference"
    )
    use_fp16: bool = Field(default=False, description="Use half-precision (FP16)")
    preprocessing: dict[str, Any] | None = Field(
        default=None, description="Preprocessing configuration"
    )


# ============================================================================
# Training Schemas
# ============================================================================


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    name: Literal["adam", "sgd", "adamw", "rmsprop"] = Field(
        default="adam", description="Optimizer name"
    )
    learning_rate: float = Field(gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay (L2 penalty)")
    momentum: float | None = Field(default=None, ge=0, le=1, description="Momentum for SGD")
    betas: tuple[float, float] | None = Field(
        default=None, description="Beta coefficients for Adam/AdamW"
    )


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    name: Literal["step", "cosine", "exponential", "reduce_on_plateau", "none"] = Field(
        default="none", description="Scheduler name"
    )
    step_size: int | None = Field(default=None, gt=0, description="Step size for StepLR")
    gamma: float | None = Field(default=None, gt=0, le=1, description="Decay factor")
    patience: int | None = Field(
        default=None, gt=0, description="Patience for ReduceLROnPlateau"
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters and settings."""

    epochs: int = Field(gt=0, description="Number of training epochs")
    batch_size: int = Field(gt=0, description="Batch size for training")
    optimizer: OptimizerConfig = Field(description="Optimizer configuration")
    scheduler: SchedulerConfig | None = Field(
        default=None, description="Learning rate scheduler configuration"
    )
    gradient_clip_value: float | None = Field(
        default=None, gt=0, description="Gradient clipping value"
    )
    early_stopping_patience: int | None = Field(
        default=None, gt=0, description="Early stopping patience"
    )
    validation_frequency: int = Field(
        default=1, gt=0, description="Validate every N epochs"
    )
    checkpoint_frequency: int = Field(
        default=1, gt=0, description="Save checkpoint every N epochs"
    )


# ============================================================================
# Experiment Tracking Schemas
# ============================================================================


class LoggerConfig(BaseModel):
    """Experiment logger configuration."""

    provider: Literal["wandb", "mlflow", "tensorboard"] = Field(
        description="Logging provider"
    )
    project_name: str = Field(description="Project name")
    experiment_name: str = Field(description="Experiment name")
    tags: list[str] | None = Field(default=None, description="Experiment tags")
    notes: str | None = Field(default=None, description="Experiment notes")
    log_frequency: int = Field(default=10, gt=0, description="Log metrics every N steps")


class MetricLog(BaseModel):
    """Generic metric log entry."""

    name: str = Field(description="Metric name")
    value: float = Field(description="Metric value")
    step: int | None = Field(default=None, ge=0, description="Training step/epoch")
    timestamp: str | None = Field(default=None, description="Log timestamp")
    split: Literal["train", "val", "test"] | None = Field(
        default=None, description="Dataset split"
    )


class ArtifactLog(BaseModel):
    """Generic artifact log entry."""

    path: str = Field(description="Artifact file path")
    artifact_type: Literal["model", "plot", "data", "config", "other"] = Field(
        description="Type of artifact"
    )
    name: str = Field(description="Artifact name")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional artifact metadata"
    )


class ExperimentMetadata(BaseModel):
    """Experiment metadata."""

    experiment_id: str = Field(description="Unique experiment identifier")
    run_name: str = Field(description="Human-readable run name")
    tags: list[str] | None = Field(default=None, description="Experiment tags")
    description: str | None = Field(default=None, description="Experiment description")
    git_commit: str | None = Field(default=None, description="Git commit hash")
    environment: dict[str, Any] | None = Field(
        default=None, description="Environment information (Python version, packages, etc.)"
    )
    start_time: str = Field(description="Experiment start timestamp")
    end_time: str | None = Field(default=None, description="Experiment end timestamp")


class ExperimentSummary(BaseModel):
    """Experiment summary with key results."""

    experiment_id: str = Field(description="Experiment identifier")
    status: Literal["running", "completed", "failed", "stopped"] = Field(
        description="Experiment status"
    )
    best_metrics: dict[str, float] = Field(description="Best metric values achieved")
    final_metrics: dict[str, float] = Field(description="Final metric values")
    total_epochs: int = Field(ge=0, description="Total epochs completed")
    total_steps: int = Field(ge=0, description="Total training steps completed")
    duration_seconds: float = Field(ge=0, description="Total experiment duration")
    artifacts: list[ArtifactLog] = Field(
        default_factory=list, description="List of logged artifacts"
    )


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""

    experiment_id: str = Field(description="Unique experiment identifier")
    training: TrainingConfig = Field(description="Training configuration")
    logger: LoggerConfig = Field(description="Logger configuration")
    model_config: dict = Field(description="Model-specific configuration")
    data_config: dict = Field(description="Data-specific configuration")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


# ============================================================================
# RL-Specific Schemas
# ============================================================================


class ReplayBufferConfig(BaseModel):
    """Replay buffer configuration."""

    capacity: int = Field(gt=0, description="Buffer capacity")
    batch_size: int = Field(gt=0, description="Sampling batch size")
    prioritized: bool = Field(default=False, description="Use prioritized replay")
    alpha: float = Field(default=0.6, ge=0, le=1, description="Prioritization exponent")
    beta: float = Field(default=0.4, ge=0, le=1, description="Importance sampling")


class RLAgentConfig(BaseModel):
    """RL agent configuration."""

    algorithm: Literal["dqn", "ppo", "sac", "a2c", "ddpg"] = Field(
        description="RL algorithm"
    )
    gamma: float = Field(default=0.99, ge=0, le=1, description="Discount factor")
    learning_rate: float = Field(gt=0, description="Learning rate")
    epsilon_start: float = Field(default=1.0, ge=0, le=1, description="Initial epsilon")
    epsilon_end: float = Field(default=0.01, ge=0, le=1, description="Final epsilon")
    epsilon_decay: float = Field(default=0.995, ge=0, le=1, description="Epsilon decay")
    target_update_freq: int = Field(
        default=1000, gt=0, description="Target network update frequency"
    )


class RLTrainingConfig(BaseModel):
    """RL training configuration."""

    total_timesteps: int = Field(gt=0, description="Total training timesteps")
    warmup_steps: int = Field(default=1000, ge=0, description="Random action warmup steps")
    train_freq: int = Field(default=4, gt=0, description="Training frequency (steps)")
    gradient_steps: int = Field(default=1, gt=0, description="Gradient steps per training")
    eval_freq: int = Field(default=10000, gt=0, description="Evaluation frequency")
    eval_episodes: int = Field(default=10, gt=0, description="Episodes per evaluation")
    save_freq: int = Field(default=10000, gt=0, description="Checkpoint save frequency")
    replay_buffer: ReplayBufferConfig = Field(description="Replay buffer config")


class RLEnvironmentConfig(BaseModel):
    """RL environment configuration."""

    env_id: str = Field(description="Environment ID (e.g., 'CartPole-v1')")
    env_type: Literal["gym", "gymnasium", "custom"] = Field(description="Environment type")
    num_envs: int = Field(default=1, gt=0, description="Number of parallel environments")
    max_episode_steps: Optional[int] = Field(default=None, description="Max steps per episode")
    render_mode: Optional[str] = Field(default=None, description="Render mode")


class RLExperimentConfig(BaseModel):
    """Full RL experiment configuration."""

    experiment_id: str = Field(description="Unique experiment ID")
    environment: RLEnvironmentConfig = Field(description="Environment config")
    agent: RLAgentConfig = Field(description="Agent config")
    training: RLTrainingConfig = Field(description="Training config")
    seed: Optional[int] = Field(default=None, description="Random seed")
