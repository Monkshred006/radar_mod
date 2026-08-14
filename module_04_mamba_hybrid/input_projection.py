"""Sensor Token & Input Projection Module for Module 4.

Converts Module 3 output representations into model-dimension embeddings [B, T, S, D_model]
and provides sensor aggregation to [B, T, D_model].
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_04_mamba_hybrid.config import MambaHybridConfig


class SensorTokenProjection(nn.Module):
    """Projects sensor-aware tokens [B, T, S, D_features] into [B, T, S, D_model].

    Preserves individual sensor identity while projecting to model dimension.
    Applies token masking so padded or missing sensors are zeroed.
    """

    def __init__(self, config: MambaHybridConfig):
        super().__init__()
        self.config = config
        self.d_features = config.sensor_feature_dim
        self.d_model = config.d_model

        # Linear projection behind modular wrapper for future low-bit replacement
        self.proj = nn.Linear(self.d_features, self.d_model)
        self.norm = nn.LayerNorm(self.d_model) if config.normalization == "layernorm" else nn.Identity()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Optional flat projection if only fused features [B, T, F_fused] are provided
        self.flat_proj = nn.Linear(config.fused_feature_dim, self.d_model)

    def forward(
        self,
        module3_output: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Process Module 3 output dictionary.

        Args:
            module3_output: Output dict from Module 3 containing 'tokens', 'token_mask',
                            and/or 'features'.

        Returns:
            Tuple of:
                - sensor_tokens: torch.Tensor [B, T, S, D_model]
                - sensor_mask: torch.Tensor [B, T, S] (boolean mask, True = valid)
                - aggregated_sequence: torch.Tensor [B, T, D_model]
        """
        if "tokens" in module3_output and module3_output["tokens"] is not None:
            tokens = module3_output["tokens"]  # [T, S, D_feat] or [B, T, S, D_feat]
            token_mask = module3_output.get("token_mask", None)

            # Ensure batch dimension: [T, S, D] -> [1, T, S, D]
            if tokens.ndim == 3:
                tokens = tokens.unsqueeze(0)
            if token_mask is not None and token_mask.ndim == 3:
                token_mask = token_mask.unsqueeze(0)

            B, T, S, D = tokens.shape

            # Project per-sensor tokens: [B, T, S, D] -> [B, T, S, D_model]
            x_proj = self.proj(tokens)
            x_proj = self.norm(x_proj)
            x_proj = self.dropout(x_proj)

            # Build group-level sensor mask [B, T, S]
            if token_mask is not None:
                # token_mask is [B, T, S, D]; group valid if ANY feature valid or all True
                if token_mask.dtype == torch.bool:
                    group_mask = token_mask.any(dim=-1)  # [B, T, S]
                else:
                    group_mask = (token_mask > 0).any(dim=-1)
                
                # Zero out projected tokens for invalid sensors
                mask_expanded = group_mask.unsqueeze(-1)  # [B, T, S, 1]
                x_proj = x_proj * mask_expanded.to(x_proj.dtype)
            else:
                group_mask = torch.ones((B, T, S), dtype=torch.bool, device=tokens.device)

            # Aggregate across sensor groups S to form [B, T, D_model]
            # Masked mean aggregation
            valid_counts = group_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, T, 1]
            aggregated = x_proj.sum(dim=2) / valid_counts.to(x_proj.dtype)  # [B, T, D_model]

            return x_proj, group_mask, aggregated

        elif "features" in module3_output and module3_output["features"] is not None:
            # Fallback for flat fused features [B, T, F_fused]
            features = module3_output["features"]
            if features.ndim == 2:
                features = features.unsqueeze(0)  # [1, T, F_fused]
            
            B, T, F_fused = features.shape
            aggregated = self.flat_proj(features)
            aggregated = self.norm(aggregated)
            aggregated = self.dropout(aggregated)

            # Create dummy single sensor group [B, T, 1, D_model]
            sensor_tokens = aggregated.unsqueeze(2)
            group_mask = torch.ones((B, T, 1), dtype=torch.bool, device=features.device)

            return sensor_tokens, group_mask, aggregated

        else:
            raise KeyError("module3_output must contain 'tokens' or 'features'")
