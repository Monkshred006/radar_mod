"""Tiny Latent-Space Diffusion Auxiliary Branch (Phase V1 Preparation).

Provides:
- `SinusoidalPosEmb`: Time-step embedding for diffusion conditioning.
- `Tiny1DUNetDenoiser`: 1D UNet-style denoiser operating on temporal latents [B, T, H].
- `DiffusionAuxiliary`: Complete auxiliary training module computing L_diffusion.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_05_training.noise_scheduler import NoiseScheduler


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional / Timestep Embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed integer timesteps tensor [B] -> [B, dim]."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Tiny1DUNetDenoiser(nn.Module):
    """Lightweight 1D UNet Denoiser operating over temporal sequence T.

    Input: [B, T, H] noisy latent + [B] timestep t
    Output: [B, T, H] reconstructed clean latent
    """

    def __init__(self, hidden_dim: int = 64, time_emb_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Timestep MLP embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, hidden_dim),
        )

        # Encoder (Downsampling / Feature extraction along T)
        self.in_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.down1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.mid_conv1 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)
        self.mid_conv2 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Denoise latent state x_t at timestep t.

        Args:
            x_t: Noisy latent tensor `[B, T, H]`.
            t: Timestep indices `[B]`.

        Returns:
            Reconstructed latent tensor `[B, T, H]`.
        """
        B, T, H = x_t.shape

        # Time conditioning vector [B, H, 1]
        t_emb = self.time_mlp(t).unsqueeze(-1)

        # Transpose to [B, H, T] for 1D convolutions
        h = x_t.transpose(1, 2)
        h = self.in_conv(h) + t_emb  # [B, H, T]
        skip = h

        # Downsample
        h_down = F.silu(self.down1(h))  # [B, 2H, T/2]

        # Bottleneck
        h_mid = F.silu(self.mid_conv1(h_down))
        h_mid = F.silu(self.mid_conv2(h_mid))

        # Upsample
        h_up = F.silu(self.up1(h_mid))  # [B, H, T]

        # Handle odd sequence length mismatch if any
        if h_up.shape[-1] != skip.shape[-1]:
            h_up = F.interpolate(h_up, size=skip.shape[-1], mode="linear", align_corners=False)

        # Concatenate skip connection & project out
        h_out = torch.cat([h_up, skip], dim=1)  # [B, 2H, T]
        out = self.out_conv(h_out)  # [B, H, T]

        return out.transpose(1, 2)  # [B, T, H]


class DiffusionAuxiliary(nn.Module):
    """Auxiliary Latent Diffusion Module for PhotonShield AI (Phase V1 Preparation).

    Attributes:
        enabled: Boolean flag whether diffusion auxiliary loss is computed.
        timesteps: Total discrete diffusion timesteps (default 10).
        noise_std: Scale of injected Gaussian noise.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        timesteps: int = 10,
        noise_std: float = 0.1,
        enabled: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        self.noise_std = noise_std
        self.enabled = enabled

        self.scheduler = NoiseScheduler(num_timesteps=timesteps)
        self.denoiser = Tiny1DUNetDenoiser(hidden_dim=hidden_dim)

    def forward(
        self,
        latent_clean: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse clean latent and reconstruct it with denoiser.

        Args:
            latent_clean: Clean latent state tensor `[B, T, H]`.
            t: Optional timestep indices `[B]`. If None, sampled uniformly.

        Returns:
            Tuple of:
                - reconstructed_latent: `[B, T, H]`
                - target_clean: `[B, T, H]`
        """
        B = latent_clean.shape[0]
        device = latent_clean.device

        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)

        # Sample noise scaled by noise_std
        noise = torch.randn_like(latent_clean) * self.noise_std
        x_noisy, _ = self.scheduler.q_sample(latent_clean, t, noise=noise)

        # Denoise
        reconstructed = self.denoiser(x_noisy, t)
        return reconstructed, latent_clean

    def compute_loss(
        self,
        latent_clean: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE diffusion reconstruction loss.

        L_diffusion = MSE(latent_clean, reconstructed_latent)
        """
        if not self.enabled:
            return torch.zeros(1, device=latent_clean.device, dtype=latent_clean.dtype)

        reconstructed, target = self.forward(latent_clean)
        return F.mse_loss(reconstructed, target)
