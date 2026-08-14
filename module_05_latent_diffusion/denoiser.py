"""Lightweight Temporal Denoiser for Conditional Radar Latent Diffusion.

Compact 2-block temporal architecture conditioning on corrupted latent state z_c and diffusion timestep t.
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed 1D timesteps [B] into [B, dim]."""
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TemporalDenoisingBlock(nn.Module):
    """Lightweight 1D temporal convolution residual block with timestep injection."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()

        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Temporal sequence tensor [B, T, hidden_dim].
            time_emb: Timestep embedding tensor [B, hidden_dim].

        Returns:
            Processed tensor [B, T, hidden_dim].
        """
        residual = x
        h = self.norm1(x)
        h = self.conv1(h.transpose(1, 2)).transpose(1, 2)
        h = self.act1(h)

        # Inject timestep conditioning
        t_inject = self.time_proj(time_emb).unsqueeze(1)  # [B, 1, hidden_dim]
        h = h + t_inject

        h = self.norm2(h)
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        h = self.act2(h)

        return residual + h


class LightweightDenoiser(nn.Module):
    """Small conditional latent diffusion denoiser for temporal radar representations."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # 1. Timestep Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Input Projection (Concatenates z_t and z_c: 64 + 64 = 128)
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
        )

        # 3. Stacked Temporal Residual Blocks
        self.blocks = nn.ModuleList([
            TemporalDenoisingBlock(hidden_dim=hidden_dim)
            for _ in range(num_blocks)
        ])

        # 4. Output Projection (Predicts noise epsilon [B, T, latent_dim=64])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        z_t: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise epsilon from noisy latent z_t, condition z_c, and timestep t.

        Args:
            z_t: Noisy latent tensor [B, T, latent_dim=64].
            condition: Corrupted latent condition z_c [B, T, latent_dim=64].
            timestep: Timestep tensor [B] (integer in 0..T_max-1).

        Returns:
            Predicted noise tensor epsilon_hat [B, T, latent_dim=64].
        """
        # Embed timestep
        t_emb = self.time_mlp(timestep)  # [B, hidden_dim]

        # Concatenate noisy latent and corrupted condition
        x = torch.cat([z_t, condition], dim=-1)  # [B, T, 128]
        h = self.input_proj(x)

        # Process through temporal blocks
        for block in self.blocks:
            h = block(h, t_emb)

        eps_pred = self.output_proj(h)  # [B, T, 64]
        return eps_pred
