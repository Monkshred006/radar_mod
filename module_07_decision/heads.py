"""Base task head abstractions and utilities for Module 7."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class BaseTaskHead(nn.Module):
    """Base class for lightweight task-specific prediction heads.

    Args:
        d_model: Input representation dimension from Module 4 representation.
        out_dim: Output dimension of the task head.
        hidden_dim: Optional hidden dimension for 2-layer MLP. If None, single Linear layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        if hidden_dim is not None and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(d_model, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Latent representation tensor [B, D_model].

        Returns:
            Output tensor [B, out_dim].
        """
        return self.net(x)
