"""DDPM & DDIM Gaussian Diffusion Scheduler for Latent State Denoising & Inpainting.

Handles forward diffusion noising (q-sample), x0 prediction, reverse posterior sampling (p-sample),
and conditional deterministic DDIM / stochastic DDPM inpainting with data consistency.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):
    """Diffusion Scheduler supporting DDPM training and DDIM conditional inpainting."""

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
        self.beta_schedule = beta_schedule

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
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]])

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
        """Add noise according to q(z_t | z_0) = sqrt(alpha_bar_t)*z_0 + sqrt(1 - alpha_bar_t)*epsilon."""
        device = original_samples.device
        t = timesteps.to(device).long()

        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod * noise)
        return noisy_samples

    def predict_z0_from_eps(
        self,
        z_t: torch.Tensor,
        eps_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Recover predicted clean latent z0_hat = (z_t - sqrt(1 - alpha_bar_t)*eps_pred) / sqrt(alpha_bar_t)."""
        device = z_t.device
        t = timesteps.to(device).long()

        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        pred_z0 = (z_t - (sqrt_one_minus_alpha_prod * eps_pred)) / torch.clamp(sqrt_alpha_prod, min=1e-6)
        return pred_z0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sample at previous timestep z_{t-1} and predicted clean latent z0_hat."""
        t = timestep
        device = sample.device

        # 1. Recover predicted z_0
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        pred_z0 = self.predict_z0_from_eps(sample, model_output, t_tensor)

        # 2. Posterior coefficients for q(z_{t-1} | z_t, z_0)
        beta_t = self.betas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]
        alpha_t = self.alphas[t]

        c1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
        c2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
        mean = (c1 * pred_z0) + (c2 * sample)

        if t > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(self.posterior_variance[t])
            prev_sample = mean + (variance * noise)
        else:
            prev_sample = pred_z0

        return prev_sample, pred_z0

    @torch.no_grad()
    def reconstruct(
        self,
        denoiser: nn.Module,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Run conditional reverse diffusion trajectory with observed-frame data consistency.

        Mathematical Inpainting Form (DDIM with Data Consistency):
            At each step t -> t_prev:
            - Denoiser predicts noise eps_pred given (z_t, z_c, mask, t).
            - Recover clean estimate: z0_hat = (z_t - sqrt(1 - alpha_bar_t)*eps_pred) / sqrt(alpha_bar_t).
            - Synthesize next step for missing frames:
              z_{t_prev}^{gen} = sqrt(alpha_bar_{t_prev})*z0_hat + sqrt(1 - alpha_bar_{t_prev})*eps_pred.
            - Project known frames:
              Deterministic: z_{t_prev}^{obs} = sqrt(alpha_bar_{t_prev}) * condition
              Stochastic:    z_{t_prev}^{obs} = q(condition, t_prev)
            - Blending: z_{t_prev} = mask * z_{t_prev}^{obs} + (1 - mask) * z_{t_prev}^{gen}.
            - At final step t=0: z_0 = mask * condition + (1 - mask) * z0_hat.

        Args:
            denoiser: Conditional denoiser network predicting eps_hat.
            condition: Corrupted / observed latent condition z_c [B, T, D].
            mask: Binary observation mask [B, T, 1] (1=observed, 0=missing). Defaults to all 1s.
            num_inference_steps: Steps for reverse trajectory.
            deterministic: If True, executes 100% deterministic DDIM reverse sampling with no random noise.

        Returns:
            Reconstructed latent tensor z_hat [B, T, D].
        """
        denoiser.eval()
        B, T, D = condition.shape
        device = condition.device

        if mask is None:
            mask = torch.ones(B, T, 1, device=device)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(-1)

        total_train_steps = self.num_train_timesteps
        inference_steps = num_inference_steps or total_train_steps

        # Timestep mapping
        if inference_steps == total_train_steps:
            timesteps = list(reversed(range(total_train_steps)))
        else:
            timesteps = [int(round(s)) for s in reversed(torch.linspace(0, total_train_steps - 1, inference_steps).tolist())]

        # 1. Initialize z_T
        t_start = timesteps[0]
        sqrt_alpha_start = self.sqrt_alphas_cumprod[t_start]
        sqrt_one_minus_start = self.sqrt_one_minus_alphas_cumprod[t_start]

        if deterministic:
            # Deterministic ODE initialization: observed component = sqrt(alpha_bar)*z_c, missing component = 0
            z_t = (mask * (sqrt_alpha_start * condition)) + ((1.0 - mask) * torch.zeros_like(condition))
        else:
            init_noise = torch.randn(B, T, D, device=device)
            t_start_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
            z_t_observed = self.add_noise(condition, init_noise, t_start_tensor)
            z_t = (mask * z_t_observed) + ((1.0 - mask) * init_noise)

        # 2. Reverse Inpainting Trajectory
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise conditioned on (z_t, z_c, mask, t)
            eps_pred = denoiser(z_t=z_t, condition=condition, timestep=t_tensor, mask=mask)

            alpha_bar_t = self.alphas_cumprod[t]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t]

            pred_z0 = (z_t - (sqrt_one_minus_t * eps_pred)) / torch.clamp(sqrt_alpha_t, min=1e-6)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                sqrt_alpha_prev = self.sqrt_alphas_cumprod[t_prev]
                sqrt_one_minus_prev = self.sqrt_one_minus_alphas_cumprod[t_prev]

                # Deterministic DDIM trajectory for generated component
                z_prev_gen = (sqrt_alpha_prev * pred_z0) + (sqrt_one_minus_prev * eps_pred)

                # Observed frame data consistency projection
                if deterministic:
                    z_prev_known = sqrt_alpha_prev * condition
                else:
                    noise_known = torch.randn(B, T, D, device=device)
                    t_prev_tensor = torch.full((B,), t_prev, device=device, dtype=torch.long)
                    z_prev_known = self.add_noise(condition, noise_known, t_prev_tensor)

                z_t = (mask * z_prev_known) + ((1.0 - mask) * z_prev_gen)
            else:
                # Final step t=0: exact observation for observed frames, pred_z0 for missing frames
                z_t = (mask * condition) + ((1.0 - mask) * pred_z0)

        return z_t
