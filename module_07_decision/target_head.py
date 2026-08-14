"""Target Indication Task Head for Module 7."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from module_07_decision.heads import BaseTaskHead
from module_07_decision.config import DecisionModelConfig


class TargetHead(BaseTaskHead):
    """Target Indication Classification Head.

    Maps latent representation pooled_output [B, D_model] to unnormalized logits [B, num_classes].
    Does NOT apply argmax or softmax inside the neural module.

    Args:
        d_model: Input dimension.
        num_classes: Number of target classification categories.
        hidden_dim: Optional hidden dimension for MLP head.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__(d_model=d_model, out_dim=num_classes, hidden_dim=hidden_dim, dropout=dropout)
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config: DecisionModelConfig) -> "TargetHead":
        """Instantiate TargetHead from DecisionModelConfig."""
        return cls(
            d_model=config.d_model,
            num_classes=config.num_target_classes,
            hidden_dim=config.target_hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized target logits.

        Args:
            x: Input tensor [B, d_model].

        Returns:
            Target logits tensor [B, num_classes].
        """
        return super().forward(x)
