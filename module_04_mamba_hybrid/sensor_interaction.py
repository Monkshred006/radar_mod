"""Cross-Sensor Interaction Branch Module for Module 4.

Models interactions between different sensor groups (optical, environment, motion, distance, quality)
across the sensor group dimension S (not across time T) using lightweight multi-head self-attention
and learned sensor group weighting.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_04_mamba_hybrid.config import MambaHybridConfig


class CrossSensorInteractionBranch(nn.Module):
    """Lightweight Multi-Head Cross-Sensor Interaction Branch.

    Operates on sensor group tokens [B, T, S, D_model].
    Reshapes sequence to [B*T, S, D_model] to perform cross-sensor self-attention
    at each timestep independently (O(S^2) per timestep, negligible for S=5).
    Uses learned group weighting to aggregate sensor tokens into temporal features.
    """

    def __init__(self, config: MambaHybridConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = self.d_model // self.num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(self.d_model) if config.normalization == "layernorm" else nn.Identity()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Learned group aggregation weights [D_model -> 1]
        self.group_weight_proj = nn.Linear(self.d_model, 1)

    def forward(
        self,
        sensor_tokens: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass across sensor group dimension.

        Args:
            sensor_tokens: Tensor [B, T, S, D_model].
            sensor_mask: Optional boolean mask [B, T, S] (True = valid, False = masked).

        Returns:
            Tuple of:
                - interacted_sensor_tokens: Tensor [B, T, S, D_model]
                - aggregated_sequence: Tensor [B, T, D_model]
        """
        B, T, S, D = sensor_tokens.shape

        if not self.config.use_sensor_attention:
            # Simple weighted aggregation if attention disabled
            logits = self.group_weight_proj(sensor_tokens)  # [B, T, S, 1]
            if sensor_mask is not None:
                mask_exp = sensor_mask.unsqueeze(-1)
                logits = logits.masked_fill(~mask_exp, -1e9)
            weights = F.softmax(logits, dim=2)
            agg = (sensor_tokens * weights).sum(dim=2)  # [B, T, D_model]
            return sensor_tokens, agg

        # Reshape to [B*T, S, D_model]
        tokens_flat = sensor_tokens.reshape(B * T, S, D)

        # Handle attention mask [B*T, S] -> key_padding_mask expected by PyTorch MHA (True = MASKED)
        key_padding_mask = None
        if sensor_mask is not None:
            mask_flat = sensor_mask.reshape(B * T, S)
            key_padding_mask = ~mask_flat  # PyTorch MHA expects True for elements to IGNORE

            # If all sensors are masked for a timestep, unmask first sensor to avoid NaN
            all_masked = key_padding_mask.all(dim=-1)
            if all_masked.any():
                key_padding_mask[all_masked, 0] = False

        # Multi-Head Self-Attention across S dimension
        tokens_norm = self.norm(tokens_flat)
        attn_out, _ = self.mha(
            query=tokens_norm,
            key=tokens_norm,
            value=tokens_norm,
            key_padding_mask=key_padding_mask,
        )

        attn_out = self.dropout(attn_out)
        interacted_flat = tokens_flat + attn_out  # Residual connection across S
        interacted = interacted_flat.reshape(B, T, S, D)

        # Learned weighted aggregation of sensor tokens S -> temporal sequence [B, T, D_model]
        group_logits = self.group_weight_proj(interacted)  # [B, T, S, 1]
        if sensor_mask is not None:
            mask_exp = sensor_mask.unsqueeze(-1)
            group_logits = group_logits.masked_fill(~mask_exp, -1e9)

        group_weights = F.softmax(group_logits, dim=2)  # [B, T, S, 1]
        aggregated = (interacted * group_weights).sum(dim=2)  # [B, T, D_model]

        return interacted, aggregated
