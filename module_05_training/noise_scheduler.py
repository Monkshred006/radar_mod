"""Noise Scheduler for Diffusion Auxiliary Branch (Phase V1 Preparation).

Provides discrete diffusion forward process schedules (linear, cosine)
and noise injection operations for latent sequence representations [B, T, H].
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class NoiseScheduler:
    """Discrete Timestep Gaussian Noise Scheduler for Latent Diffusion.

    Calculates:
    - beta_t: Variance schedule
    - alpha_t = 1 - beta_t
    - alpha_bar_t = cumprod(alpha_t)
    - q_sample: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    """

    def __init__(
        self,
        num_timesteps: int = 10,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.num_timesteps = num_timesteps
        self.schedule = schedule

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / (1 + 0.008) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device: Union[str, torch.device]) -> "NoiseScheduler":
        """Move scheduler tensors to target device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse data (forward process) at timestep t.

        Args:
            x_0: Clean latent tensor `[B, T, H]`.
            t: Timesteps tensor `[B]` with integer values in `[0, num_timesteps - 1]`.
            noise: Optional pre-generated noise tensor `[B, T, H]`.

        Returns:
            Tuple of:
                - x_t: Noisy latent tensor `[B, T, H]`
                - noise: Injected Gaussian noise `[B, T, H]`
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Move scheduler constants to device of x_0 if needed
        if self.sqrt_alphas_cumprod.device != x_0.device:
            self.to(x_0.device)

        # Extract coefficients for batch
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)  # [B, 1, 1]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)  # [B, 1, 1]

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
