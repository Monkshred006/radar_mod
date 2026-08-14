"""Temporal & Positional Encoding Module for Module 4.

Provides learned temporal embeddings or timestamp-aware relative encoding for sequence inputs.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from module_04_mamba_hybrid.config import MambaHybridConfig


class TemporalEncoding(nn.Module):
    """Adds temporal encoding to sequence representations [B, T, D_model].

    Supports:
    - 'learned': Standard learned positional embedding per sequence step.
    - 'timestamp_delta': Encodes relative physical timestamp differences dt = t[i] - t[i-1].
    - 'none': Identity operation.
    """

    def __init__(self, config: MambaHybridConfig):
        super().__init__()
        self.config = config
        self.encoding_type = config.temporal_encoding_type
        self.d_model = config.d_model
        self.max_len = config.max_sequence_length

        if self.encoding_type == "learned" and config.use_temporal_encoding:
            self.pos_embed = nn.Embedding(self.max_len, self.d_model)
        elif self.encoding_type == "timestamp_delta" and config.use_temporal_encoding:
            # Linear projection of relative time delta dt into d_model
            self.dt_proj = nn.Sequential(
                nn.Linear(1, self.d_model // 2),
                nn.SiLU(),
                nn.Linear(self.d_model // 2, self.d_model),
            )
        else:
            self.pos_embed = None

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply temporal encoding to input x [B, T, D_model].

        Args:
            x: Input sequence tensor [B, T, D_model].
            timestamps: Optional timestamp tensor [T] or [B, T].

        Returns:
            Encoded tensor [B, T, D_model].
        """
        if not self.config.use_temporal_encoding or self.encoding_type == "none":
            return x

        B, T, D = x.shape

        if self.encoding_type == "learned":
            # Truncate position indices to max_len
            positions = torch.arange(T, device=x.device).clamp(max=self.max_len - 1)  # [T]
            pos_emb = self.pos_embed(positions).unsqueeze(0)  # [1, T, D_model]
            return x + pos_emb

        elif self.encoding_type == "timestamp_delta" and timestamps is not None:
            if timestamps.ndim == 1:
                ts = timestamps.unsqueeze(0).expand(B, -1)  # [B, T]
            else:
                ts = timestamps

            # Compute dt = t_i - t_{i-1}, with dt[0] = 0
            dt = torch.zeros_like(ts)
            dt[:, 1:] = ts[:, 1:] - ts[:, :-1]
            dt = dt.unsqueeze(-1).to(dtype=x.dtype)  # [B, T, 1]

            # Scale dt by median timestep for stability
            dt_scaled = dt / (dt.median().clamp(min=1e-6))
            dt_emb = self.dt_proj(dt_scaled)  # [B, T, D_model]
            return x + dt_emb

        return x
