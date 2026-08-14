"""Hybrid Block Module for Module 4.

Combines Mamba Temporal SSM, Cross-Sensor Interaction, Pre-LayerNorm,
Residual Connections, and Feed-Forward Network into a unified block.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.normalization import get_normalization_layer
from module_04_mamba_hybrid.mamba_block import MambaTemporalBranch
from module_04_mamba_hybrid.sensor_interaction import CrossSensorInteractionBranch


class FeedForwardNetwork(nn.Module):
    """Configurable Feed-Forward Network (FFN).

    D_model -> Linear(d_model * ffn_mult) -> GELU -> Dropout -> Linear(d_model).
    Linear layers are modularized for future BitNet low-bit replacement.
    """

    def __init__(self, config: MambaHybridConfig):
        super().__init__()
        self.d_model = config.d_model
        self.hidden_dim = config.d_model * config.ffn_multiplier

        self.fc1 = nn.Linear(self.d_model, self.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(self.hidden_dim, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        return self.fc2(h)


class HybridBlock(nn.Module):
    """PhotonShield Hybrid Mamba-Interaction Block.

    Architecture (Pre-LN):
    x
    ├── Norm1 -> Mamba Temporal Branch -> + Residual
    ├── Norm2 -> Sensor Interaction Branch -> + Residual
    └── Norm3 -> FFN -> + Residual
    """

    def __init__(self, config: MambaHybridConfig):
        super().__init__()
        self.config = config

        # 1. Normalization layers
        self.norm1 = get_normalization_layer(config.d_model, config.normalization)
        self.norm2 = get_normalization_layer(config.d_model, config.normalization)
        self.norm3 = get_normalization_layer(config.d_model, config.normalization)

        # 2. Branches
        self.mamba_branch = MambaTemporalBranch(config)
        self.sensor_branch = CrossSensorInteractionBranch(config)
        self.ffn = FeedForwardNetwork(config)

    def forward(
        self,
        x: torch.Tensor,
        sensor_tokens: Optional[torch.Tensor] = None,
        sensor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for single Hybrid Block.

        Args:
            x: Temporal sequence tensor [B, T, D_model].
            sensor_tokens: Optional per-sensor tokens [B, T, S, D_model].
            sensor_mask: Optional boolean sensor mask [B, T, S].

        Returns:
            Tuple of:
                - Output temporal sequence [B, T, D_model]
                - Updated per-sensor tokens [B, T, S, D_model]
        """
        # 1. Mamba Temporal Branch + Residual
        if self.config.use_mamba:
            mamba_in = self.norm1(x)
            mamba_out = self.mamba_branch(mamba_in)
            x = x + mamba_out

        # 2. Sensor Interaction Branch + Residual
        if self.config.use_sensor_attention and sensor_tokens is not None:
            updated_sensor_tokens, sensor_agg = self.sensor_branch(
                sensor_tokens, sensor_mask=sensor_mask
            )
            # Add cross-sensor aggregated features into temporal sequence
            sensor_in = self.norm2(sensor_agg)
            x = x + sensor_in
        else:
            updated_sensor_tokens = sensor_tokens

        # 3. Feed-Forward Network + Residual
        ffn_in = self.norm3(x)
        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out

        return x, updated_sensor_tokens
