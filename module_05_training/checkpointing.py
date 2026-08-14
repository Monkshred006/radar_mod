"""Checkpointing utilities for Module 5 training.

Saves and restores complete training state including:
- model state_dict
- optimizer state_dict
- scheduler state_dict
- epoch
- best validation metric
- training configuration
- model configuration
- random seed / RNG state
- training history
- split information
"""

from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


def _serialise_value(v: Any) -> Any:
    """Recursively convert non-JSON-serialisable values."""
    if isinstance(v, Path):
        return str(v)
    if is_dataclass(v) and not isinstance(v, type):
        return asdict(v)
    if isinstance(v, dict):
        return {str(k): _serialise_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialise_value(i) for i in v]
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return str(v)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_metric: float,
    training_config: Any,
    model_config: Any,
    history: List[Dict[str, Any]],
    seed: int,
    split_info: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a complete training checkpoint.

    Args:
        path: Destination file path (e.g. "checkpoints/best.pt").
        model: The nn.Module to save.
        optimizer: Current optimizer.
        scheduler: Current LR scheduler (or None).
        epoch: Current epoch number.
        best_val_metric: Best monitored validation metric seen so far.
        training_config: TrainingConfig dataclass instance.
        model_config: Model config dataclass instance (e.g. MambaHybridConfig).
        history: List of per-epoch metric dicts.
        seed: Random seed used for this run.
        split_info: Optional scene-level split manifest.
        extra: Optional additional dict to include in the checkpoint.
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    sched_state = None
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        sched_state = scheduler.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": sched_state,
        "best_val_metric": best_val_metric,
        "training_config": _serialise_value(training_config),
        "model_config": _serialise_value(model_config),
        "history": history,
        "seed": seed,
        "split_info": split_info or {},
        "extra": extra or {},
    }

    torch.save(checkpoint, ckpt_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: nn.Module to restore parameters into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Device to map tensors to.

    Returns:
        Dict of the raw checkpoint (for access to epoch, history, config, etc.)
    """
    map_location = device or "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
