"""Checkpointing Utilities for Module 4.

Provides save and load functionality for model weights, configuration, optimizer states,
training metrics, and random seeds.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn

from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    config: Optional[MambaHybridConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> None:
    """Save complete model checkpoint.

    Args:
        filepath: Target filepath (.pt or .pth).
        model: PhotonMambaHybrid or wrapped PyTorch model instance.
        config: MambaHybridConfig instance.
        optimizer: Optional optimizer instance.
        scheduler: Optional learning rate scheduler instance.
        epoch: Current training epoch.
        metrics: Optional dictionary of evaluation metrics.
        seed: Optional random seed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Unwrap DDP / DataParallel if present
    raw_model = model.module if hasattr(model, "module") else model
    cfg = config or getattr(raw_model, "config", None)

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "metrics": metrics or {},
        "seed": seed,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        filepath: Source checkpoint filepath.
        model: Model instance to populate.
        optimizer: Optional optimizer instance to restore.
        scheduler: Optional scheduler instance to restore.
        strict: Whether state_dict matching must be strict.

    Returns:
        Checkpoint dictionary containing loaded metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
