"""Configurable Loss Functions for Module 4.

Provides standard loss functions and factory utilities for training and evaluation.
"""

from __future__ import annotations
from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Combined Classification (CrossEntropy) and Regression (MSE) Loss."""

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            predictions: Dict containing 'logits' and 'regression'.
            targets: Dict containing 'target_class' and 'target_regression'.

        Returns:
            Scalar loss tensor.
        """
        loss_cls = self.cls_loss(predictions["logits"], targets["target_class"])
        loss_reg = self.reg_loss(predictions["regression"], targets["target_regression"])
        return self.cls_weight * loss_cls + self.reg_weight * loss_reg


def get_loss_fn(loss_name: str, **kwargs: Any) -> nn.Module:
    """Factory function for loss functions.

    Args:
        loss_name: Name of loss ("cross_entropy", "bce", "mse", "l1", "smooth_l1", "multitask").

    Returns:
        nn.Module loss instance.
    """
    name = loss_name.lower().replace("-", "_")

    if name in ("cross_entropy", "crossentropy", "ce"):
        return nn.CrossEntropyLoss(**kwargs)
    elif name in ("bce", "bce_with_logits"):
        return nn.BCEWithLogitsLoss(**kwargs)
    elif name in ("mse", "mean_squared_error"):
        return nn.MSELoss(**kwargs)
    elif name in ("l1", "mean_absolute_error"):
        return nn.L1Loss(**kwargs)
    elif name in ("smooth_l1", "huber"):
        return nn.SmoothL1Loss(**kwargs)
    elif name == "multitask":
        return MultiTaskLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
