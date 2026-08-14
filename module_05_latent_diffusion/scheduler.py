"""DDPM Noise Scheduler for Latent State Diffusion.

Handles forward diffusion noising (q-sample) and reverse iterative denoising (p-sample).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):
    """DDPM Gaussian Diffusion Scheduler for temporal latent states."""

    def __init__(
        self,
        num_train_timesteps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Calculations for diffusion q(z_t | z_0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(z_{t-1} | z_t, z_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", torch.clamp(posterior_variance, min=1e-20))
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latent samples according to forward diffusion schedule.

        Args:
            original_samples: Clean latent tensor z_0 [B, T, D].
            noise: Random Gaussian noise epsilon [B, T, D].
            timesteps: 1D tensor of timesteps [B].

        Returns:
            Noised latent tensor z_t [B, T, D].
        """
        device = original_samples.device
        t = timesteps.to(device).long()

        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod * noise)
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the sample at the previous timestep z_{t-1} from z_t and predicted noise epsilon."""
        t = timestep
        device = sample.device

        # 1. Predict z_0 from model_output (predicted epsilon)
        sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas[t])
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t]
        beta_t = self.betas[t]

        pred_z0 = (sample - (sqrt_one_minus_alpha_prod * model_output)) / self.sqrt_alphas_cumprod[t]
        
        # 2. Compute posterior mean
        c1 = (torch.sqrt(self.alphas_cumprod_prev[t]) * beta_t) / (1.0 - self.alphas_cumprod[t])
        c2 = (torch.sqrt(self.alphas[t]) * (1.0 - self.alphas_cumprod_prev[t])) / (1.0 - self.alphas_cumprod[t])
        mean = (c1 * pred_z0) + (c2 * sample)

        if t > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(self.posterior_variance[t])
            prev_sample = mean + (variance * noise)
        else:
            prev_sample = mean

        return prev_sample

    @torch.no_grad()
    def reconstruct(
        self,
        denoiser: nn.Module,
        condition: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run full reverse diffusion trajectory conditioned on corrupted latent z_c [B, T, D].

        Args:
            denoiser: Neural network predicting epsilon_hat given (z_t, z_c, t).
            condition: Corrupted latent state z_c [B, T, D].
            num_inference_steps: Number of denoising steps (defaults to num_train_timesteps).

        Returns:
            Reconstructed clean latent tensor z_hat [B, T, D].
        """
        denoiser.eval()
        B, T, D = condition.shape
        device = condition.device

        total_steps = num_inference_steps or self.num_train_timesteps
        
        # Start from pure standard Gaussian noise in latent space
        z_t = torch.randn(B, T, D, device=device)

        for step_idx in reversed(range(total_steps)):
            t_tensor = torch.full((B,), step_idx, device=device, dtype=torch.long)
            eps_pred = denoiser(z_t=z_t, condition=condition, timestep=t_tensor)
            z_t = self.step(model_output=eps_pred, timestep=step_idx, sample=z_t)

        return z_t
