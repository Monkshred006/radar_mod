"""Checkpointing utilities for Module 7 Decision Heads and Configs."""

from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Dict, Any, Optional
import torch

from module_07_decision.config import DecisionModelConfig, DecisionConfig
from module_07_decision.multitask import PhotonShieldMultiTask


def save_decision_checkpoint(
    path: str,
    multi_task_model: PhotonShieldMultiTask,
    model_config: DecisionModelConfig,
    decision_config: DecisionConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 1,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """Save Module 7 Decision heads and configurations checkpoint.

    Args:
        path: Path to output checkpoint file (.pt).
        multi_task_model: PhotonShieldMultiTask module instance.
        model_config: DecisionModelConfig.
        decision_config: DecisionConfig.
        optimizer: Optional optimizer.
        epoch: Training epoch count.
        metrics: Optional dictionary of evaluation metrics.

    Returns:
        Absolute path to saved checkpoint file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "module": "module_07_decision",
        "model_state_dict": multi_task_model.state_dict(),
        "model_config": dataclasses.asdict(model_config),
        "decision_config": dataclasses.asdict(decision_config),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "metrics": metrics or {},
    }

    torch.save(payload, str(out_path))
    return str(out_path.absolute())


def load_decision_checkpoint(
    path: str,
    multi_task_model: PhotonShieldMultiTask,
) -> Dict[str, Any]:
    """Load state dict into a PhotonShieldMultiTask model and return raw payload.

    Args:
        path: Path to saved Module 7 checkpoint.
        multi_task_model: Target PhotonShieldMultiTask module.

    Returns:
        Loaded checkpoint raw payload dictionary.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    multi_task_model.load_state_dict(payload["model_state_dict"])
    return payload
