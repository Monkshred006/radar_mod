"""Loss functions for Module 5 FP32 training.

Extends Module 4's loss utilities with training-aware wrappers,
NaN/Inf detection, and weighted multi-task loss support.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from module_05_training.config import TrainingConfig


class TrainingNaNError(RuntimeError):
    """Raised when loss or gradients become NaN or Inf during training.

    Training must fail loudly rather than silently continue with corrupted values.
    """


class SafeLoss(nn.Module):
    """Wraps any nn.Module loss to add NaN/Inf detection."""

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(prediction, target)
        if not torch.isfinite(loss):
            raise TrainingNaNError(
                f"Loss is not finite: {loss.item():.6f}. "
                "Check input data, targets, learning rate, and model outputs."
            )
        return loss


class WeightedMultiTaskLoss(nn.Module):
    """Weighted sum of per-task losses.

    L_total = Σ λ_i * L_task_i

    Task weights are configurable. If a task's weight is 0, that task's
    loss is computed but not added to the total (still logged).
    """

    def __init__(
        self,
        task_losses: Dict[str, nn.Module],
        task_weights: Dict[str, float],
    ):
        super().__init__()
        self.task_losses = nn.ModuleDict(task_losses)
        self.task_weights = task_weights

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted total loss and per-task loss components.

        Args:
            predictions: {task_name: prediction_tensor}
            targets: {task_name: target_tensor}

        Returns:
            (total_loss, per_task_losses dict)
        """
        device = next(iter(predictions.values())).device
        total = torch.zeros(1, device=device, dtype=torch.float32)
        per_task: Dict[str, torch.Tensor] = {}

        for task_name, loss_fn in self.task_losses.items():
            if task_name not in predictions or task_name not in targets:
                continue
            t_loss = loss_fn(predictions[task_name], targets[task_name])
            per_task[task_name] = t_loss
            weight = self.task_weights.get(task_name, 1.0)
            total = total + weight * t_loss

        if not torch.isfinite(total):
            raise TrainingNaNError(
                f"Multi-task total loss is not finite: {total.item():.6f}."
            )
        return total.squeeze(), per_task


def get_loss_fn(config: TrainingConfig) -> nn.Module:
    """Factory: create the configured loss function wrapped with NaN detection.

    Args:
        config: TrainingConfig specifying loss_name and target_type.

    Returns:
        nn.Module loss function.
    """
    name = config.loss_name.lower()
    kwargs = config.loss_kwargs or {}

    if name == "cross_entropy" or name == "crossentropy":
        base = nn.CrossEntropyLoss(**kwargs)
    elif name == "bce_with_logits" or name == "bce":
        base = nn.BCEWithLogitsLoss(**kwargs)
    elif name == "mse":
        base = nn.MSELoss(**kwargs)
    elif name == "l1" or name == "mae":
        base = nn.L1Loss(**kwargs)
    elif name == "smooth_l1" or name == "huber":
        base = nn.SmoothL1Loss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss: '{name}'. "
            "Choose from: cross_entropy, bce_with_logits, mse, l1, smooth_l1"
        )

    return SafeLoss(base)
