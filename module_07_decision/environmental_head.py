"""Environmental Assessment Task Head for Module 7."""

from __future__ import annotations
from typing import Optional, Literal
import torch
import torch.nn as nn

from module_07_decision.heads import BaseTaskHead
from module_07_decision.config import DecisionModelConfig


class EnvironmentalHead(BaseTaskHead):
    """Environmental Assessment Head.

    Supports dual modes:
      1. Regression mode: outputs continuous values [B, num_outputs] (e.g., temp, humidity, pressure).
      2. Classification mode: outputs environmental category logits [B, num_classes].

    Args:
        d_model: Input dimension.
        mode: 'regression' or 'classification'.
        num_outputs: Number of continuous regression outputs (if mode=='regression').
        num_classes: Number of discrete environmental categories (if mode=='classification').
        hidden_dim: Optional hidden dimension for MLP head.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        mode: Literal["regression", "classification"] = "regression",
        num_outputs: int = 3,
        num_classes: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        self.mode = mode
        out_dim = num_outputs if mode == "regression" else num_classes
        super().__init__(d_model=d_model, out_dim=out_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.num_outputs = num_outputs
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config: DecisionModelConfig) -> "EnvironmentalHead":
        """Instantiate EnvironmentalHead from DecisionModelConfig."""
        return cls(
            d_model=config.d_model,
            mode=config.environment_mode,
            num_outputs=config.num_environment_outputs,
            num_classes=config.num_environment_classes,
            hidden_dim=config.environment_hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, d_model].

        Returns:
            Regression values tensor [B, num_outputs] or classification logits tensor [B, num_classes].
        """
        return super().forward(x)
