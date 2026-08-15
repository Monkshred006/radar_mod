"""Physics-Guided Diffusion Sampler for Adaptive Physics Inpainting."""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_05_latent_diffusion.scheduler import DDPMScheduler
from module_06_physics.physics_losses import RadarPhysicsLoss


class PhysicsGuidedSampler:
    """Performs conditional reverse diffusion inpainting with controllable test-time physics guidance."""

    def __init__(
        self,
        scheduler: DDPMScheduler,
        physics_loss_module: RadarPhysicsLoss,
    ) -> None:
        self.scheduler = scheduler
        self.physics_loss = physics_loss_module

    def reconstruct_with_guidance(
        self,
        denoiser: nn.Module,
        condition: torch.Tensor,
        mask: torch.Tensor,
        lambda_guidance: float = 0.0,
        num_inference_steps: int = 50,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Run reverse diffusion trajectory with optional physics guidance.

        When lambda_guidance = 0.0, executes standard unguided deterministic DDIM inpainting.
        When lambda_guidance > 0.0, applies differentiable physics regularization gradient to missing frames.
        """
        denoiser.eval()
        self.physics_loss.eval()

        B, T, D = condition.shape
        device = condition.device

        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)

        missing_mask = (1.0 - mask)

        total_train_steps = self.scheduler.num_train_timesteps
        inference_steps = num_inference_steps or total_train_steps

        if inference_steps == total_train_steps:
            timesteps = list(reversed(range(total_train_steps)))
        else:
            timesteps = [int(round(s)) for s in reversed(torch.linspace(0, total_train_steps - 1, inference_steps).tolist())]

        t_start = timesteps[0]
        sqrt_alpha_start = self.scheduler.sqrt_alphas_cumprod[t_start]

        if deterministic:
            z_t = (mask * (sqrt_alpha_start * condition)) + (missing_mask * torch.zeros_like(condition))
        else:
            init_noise = torch.randn(B, T, D, device=device)
            t_start_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
            z_t_observed = self.scheduler.add_noise(condition, init_noise, t_start_tensor)
            z_t = (mask * z_t_observed) + (missing_mask * init_noise)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            with torch.no_grad():
                eps_pred = denoiser(z_t=z_t, condition=condition, timestep=t_tensor, mask=mask)

            alpha_bar_t = self.scheduler.alphas_cumprod[t]
            sqrt_alpha_t = self.scheduler.sqrt_alphas_cumprod[t]
            sqrt_one_minus_t = self.scheduler.sqrt_one_minus_alphas_cumprod[t]

            pred_z0 = (z_t - (sqrt_one_minus_t * eps_pred)) / torch.clamp(sqrt_alpha_t, min=1e-6)

            # Apply Test-Time Physics Guidance on missing frames
            if lambda_guidance > 0.0 and i < len(timesteps) - 1:
                pred_z0_guided = pred_z0.detach().clone().requires_grad_(True)
                p_loss, _ = self.physics_loss(pred_z0_guided)
                grad_phys = torch.autograd.grad(p_loss, pred_z0_guided, retain_graph=False)[0]
                # Scale gradient by noise schedule variance
                step_size = lambda_guidance * float(sqrt_one_minus_t.item())
                pred_z0 = pred_z0.detach() - step_size * (grad_phys * missing_mask)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                sqrt_alpha_prev = self.scheduler.sqrt_alphas_cumprod[t_prev]
                sqrt_one_minus_prev = self.scheduler.sqrt_one_minus_alphas_cumprod[t_prev]

                z_prev_gen = (sqrt_alpha_prev * pred_z0) + (sqrt_one_minus_prev * eps_pred)

                if deterministic:
                    z_prev_known = sqrt_alpha_prev * condition
                else:
                    noise_known = torch.randn(B, T, D, device=device)
                    t_prev_tensor = torch.full((B,), t_prev, device=device, dtype=torch.long)
                    z_prev_known = self.scheduler.add_noise(condition, noise_known, t_prev_tensor)

                z_t = (mask * z_prev_known) + (missing_mask * z_prev_gen)
            else:
                z_t = (mask * condition) + (missing_mask * pred_z0)

        # Final post-refinement step if lambda > 0
        if lambda_guidance > 0.0:
            z_t_refined = z_t.detach().clone().requires_grad_(True)
            p_loss, _ = self.physics_loss(z_t_refined)
            grad_final = torch.autograd.grad(p_loss, z_t_refined, retain_graph=False)[0]
            z_t = z_t.detach() - (lambda_guidance * 0.1) * (grad_final * missing_mask)
            z_t = (mask * condition) + (missing_mask * z_t)

        return z_t.detach()
