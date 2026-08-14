"""Checkpointing utilities for Module 6 BitNet models."""

from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from module_06_bitnet.config import BitNetConfig


def _serialise(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if is_dataclass(v) and not isinstance(v, type):
        return asdict(v)
    if isinstance(v, dict):
        return {str(k): _serialise(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialise(i) for i in v]
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return str(v)


def save_bitnet_checkpoint(
    path: str,
    engine: nn.Module,
    head: nn.Module,
    bitnet_config: BitNetConfig,
    model_config: Any,
    training_config: Any,
    source_fp32_checkpoint: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    layer_stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a BitNet checkpoint.

    Args:
        path: Output file path.
        engine: BitNet Module 4 engine.
        head: BitNet task head.
        bitnet_config: BitNetConfig.
        model_config: MambaHybridConfig.
        training_config: TrainingConfig.
        source_fp32_checkpoint: Path / identifier of source FP32 checkpoint.
        optimizer: Optional optimizer state.
        scheduler: Optional scheduler state.
        epoch: Epoch index.
        metrics: Monitored metric evaluation results.
        layer_stats: Quantization layer stats dictionary.
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    combined = nn.ModuleList([engine, head])

    sched_state = None
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        sched_state = scheduler.state_dict()

    opt_state = optimizer.state_dict() if optimizer is not None else None

    checkpoint = {
        "model_state_dict": combined.state_dict(),
        "optimizer_state_dict": opt_state,
        "scheduler_state_dict": sched_state,
        "epoch": epoch,
        "bitnet_config": _serialise(bitnet_config),
        "model_config": _serialise(model_config),
        "training_config": _serialise(training_config),
        "source_fp32_checkpoint": source_fp32_checkpoint,
        "metrics": metrics or {},
        "layer_stats": layer_stats or {},
    }

    torch.save(checkpoint, ckpt_path)


def load_bitnet_checkpoint(
    path: str,
    engine: nn.Module,
    head: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a BitNet checkpoint into engine and head modules.

    Args:
        path: Path to BitNet checkpoint file.
        engine: Engine module to load parameters into.
        head: Task head module to load parameters into.
        optimizer: Optional optimizer.
        scheduler: Optional scheduler.
        device: Device to map tensors to.

    Returns:
        Raw checkpoint dictionary.
    """
    map_loc = device or "cpu"
    checkpoint = torch.load(path, map_location=map_loc, weights_only=False)

    combined = nn.ModuleList([engine, head])
    combined.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
