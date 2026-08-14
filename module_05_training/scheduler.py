"""Learning-rate scheduler factory for Module 5."""

from __future__ import annotations
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LambdaLR,
    _LRScheduler,
)

from module_05_training.config import TrainingConfig


class NoScheduler(LambdaLR):
    """No-op scheduler that keeps the learning rate constant."""

    def __init__(self, optimizer: Optimizer):
        super().__init__(optimizer, lr_lambda=lambda epoch: 1.0)


def get_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    num_batches_per_epoch: int = 1,
) -> Optional[object]:
    """Create a learning-rate scheduler from TrainingConfig.

    Args:
        optimizer: The optimizer to attach the scheduler to.
        config: TrainingConfig specifying scheduler type and hyperparameters.
        num_batches_per_epoch: Total number of batches per epoch (used for cosine).

    Returns:
        A PyTorch LR scheduler, or None if config.scheduler == "none".
    """
    name = config.scheduler.lower()
    total_steps = config.epochs * max(1, num_batches_per_epoch)

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.scheduler_min_lr,
        )
    elif name == "step":
        return StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=config.early_stopping_mode,
            patience=max(1, config.early_stopping_patience // 2),
            factor=config.scheduler_gamma,
            min_lr=config.scheduler_min_lr,
        )
    elif name == "none":
        return NoScheduler(optimizer)
    else:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            "Choose from: cosine, step, plateau, none"
        )
