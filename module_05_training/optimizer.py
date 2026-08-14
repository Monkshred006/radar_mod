"""Optimizer factory for Module 5."""

from __future__ import annotations
from typing import Iterator
import torch
import torch.nn as nn
from torch.optim import Optimizer

from module_05_training.config import TrainingConfig


def get_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Create an optimizer for the given model from TrainingConfig.

    Args:
        model: nn.Module whose parameters will be optimized.
        config: TrainingConfig specifying optimizer type and hyperparameters.

    Returns:
        A PyTorch Optimizer instance.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    name = config.optimizer.lower()

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif name == "adam":
        return torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.sgd_momentum,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: '{name}'. Choose from: adamw, adam, sgd"
        )
