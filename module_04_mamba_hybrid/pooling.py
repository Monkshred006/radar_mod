"""Sequence Pooling Modules for Module 4.

Supports:
1. `MaskedMeanPooling`: Weighted mean across non-padded sequence timesteps.
2. `MeanPooling`: Unweighted mean across timesteps T.
3. `LastStepPooling`: Selects the final valid timestep representation.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from module_04_mamba_hybrid.config import MambaHybridConfig


class SequencePooling(nn.Module):
    """Sequence Pooling Layer mapping [B, T, D_model] -> [B, D_model]."""

    def __init__(self, config: MambaHybridConfig):
        super().__init__()
        self.config = config
        self.pooling_type = config.pooling

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply sequence pooling.

        Args:
            x: Input sequence tensor [B, T, D_model].
            seq_mask: Optional sequence mask [B, T] (True = valid, False = padded).

        Returns:
            Pooled tensor [B, D_model].
        """
        B, T, D = x.shape

        if self.pooling_type == "masked_mean" and seq_mask is not None:
            mask_exp = seq_mask.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
            masked_x = x * mask_exp
            denom = seq_mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)  # [B, 1]
            return masked_x.sum(dim=1) / denom

        elif self.pooling_type == "last":
            if seq_mask is not None:
                # Find last valid index per batch element
                # seq_mask is [B, T] bool
                lengths = seq_mask.sum(dim=1) - 1  # [B] 0-indexed last valid index
                lengths = lengths.clamp(min=0)
                batch_indices = torch.arange(B, device=x.device)
                return x[batch_indices, lengths, :]
            else:
                return x[:, -1, :]

        else:  # "mean" or fallback
            return x.mean(dim=1)
