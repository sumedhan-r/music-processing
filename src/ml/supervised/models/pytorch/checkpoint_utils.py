"""PyTorch-specific checkpoint utilities."""

import torch
from pathlib import Path
from typing import Any

from src.ml.supervised.models.pytorch.base_model import PyTorchBaseModel


def save_checkpoint(
    model: PyTorchBaseModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    checkpoint_path: str | Path,
    scheduler: Any | None = None,
) -> None:
    """Save a complete training checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "model_config": model.get_config(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: PyTorchBaseModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state

    Returns:
        Dictionary containing epoch, metrics, and config
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint["metrics"],
        "model_config": checkpoint.get("model_config", {}),
    }


def export_to_onnx(
    model: PyTorchBaseModel,
    dummy_input: torch.Tensor,
    output_path: str | Path,
    **kwargs: Any,
) -> None:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        dummy_input: Example input tensor for tracing
        output_path: Path to save ONNX file
        **kwargs: Additional arguments for torch.onnx.export
    """
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=kwargs.get("opset_version", 11),
        do_constant_folding=True,
        input_names=kwargs.get("input_names", ["input"]),
        output_names=kwargs.get("output_names", ["output"]),
        dynamic_axes=kwargs.get("dynamic_axes", None),
    )
