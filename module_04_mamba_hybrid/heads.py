"""Task-Specific Prediction Heads for Module 4.

Supports:
1. `ClassificationHead`: Multi-class / binary classification from latent vectors [B, D_model].
2. `RegressionHead`: Continuous value prediction from latent vectors [B, D_model].
3. `MultiTaskHead`: Combined classification and regression from shared latent vectors [B, D_model].
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from module_04_mamba_hybrid.config import TaskHeadConfig


class ClassificationHead(nn.Module):
    """Linear / MLP Classification Head."""

    def __init__(self, d_model: int, config: Optional[TaskHeadConfig] = None):
        super().__init__()
        config = config or TaskHeadConfig()
        self.d_model = d_model
        self.num_classes = config.num_classes
        hidden_dim = config.hidden_dim or (d_model // 2)

        if config.hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim, self.num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
                nn.Linear(d_model, self.num_classes),
            )

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pooled_output: Tensor [B, D_model].

        Returns:
            Logits tensor [B, num_classes].
        """
        return self.net(pooled_output)


class RegressionHead(nn.Module):
    """Linear / MLP Regression Head."""

    def __init__(self, d_model: int, config: Optional[TaskHeadConfig] = None):
        super().__init__()
        config = config or TaskHeadConfig()
        self.d_model = d_model
        self.num_outputs = config.num_regression_outputs
        hidden_dim = config.hidden_dim or (d_model // 2)

        if config.hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim, self.num_outputs),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
                nn.Linear(d_model, self.num_outputs),
            )

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pooled_output: Tensor [B, D_model].

        Returns:
            Predictions tensor [B, num_outputs].
        """
        return self.net(pooled_output)


class MultiTaskHead(nn.Module):
    """Multi-Task Head combining classification and regression."""

    def __init__(self, d_model: int, config: Optional[TaskHeadConfig] = None):
        super().__init__()
        config = config or TaskHeadConfig()
        self.cls_head = ClassificationHead(d_model, config)
        self.reg_head = RegressionHead(d_model, config)

    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pooled_output: Tensor [B, D_model].

        Returns:
            Dict containing 'logits' [B, num_classes] and 'regression' [B, num_outputs].
        """
        return {
            "logits": self.cls_head(pooled_output),
            "regression": self.reg_head(pooled_output),
        }
