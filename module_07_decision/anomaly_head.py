"""Anomaly Detection Task Head for Module 7."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from module_07_decision.heads import BaseTaskHead
from module_07_decision.config import DecisionModelConfig


class AnomalyHead(BaseTaskHead):
    """Anomaly Detection Head.

    Maps latent representation pooled_output [B, D_model] to binary anomaly logits [B, 1].

    Args:
        d_model: Input dimension.
        hidden_dim: Optional hidden dimension for MLP head.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__(d_model=d_model, out_dim=1, hidden_dim=hidden_dim, dropout=dropout)

    @classmethod
    def from_config(cls, config: DecisionModelConfig) -> "AnomalyHead":
        """Instantiate AnomalyHead from DecisionModelConfig."""
        return cls(
            d_model=config.d_model,
            hidden_dim=config.anomaly_hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw binary anomaly logits.

        Args:
            x: Input tensor [B, d_model].

        Returns:
            Anomaly logits tensor [B, 1].
        """
        return super().forward(x)
