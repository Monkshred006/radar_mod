"""Loss functions for Conditional Latent Diffusion Training & Imputation Evaluation."""

from __future__ import annotations

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Calculates composite loss with SNR-bounded weighting:
    L_total = lambda_diff * L_diff + lambda_recon * L_recon_weighted + lambda_missing * L_missing_weighted
    """

    def __init__(
        self,
        lambda_diff: float = 1.0,
        lambda_recon: float = 0.5,
        lambda_missing: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_diff = float(lambda_diff)
        self.lambda_recon = float(lambda_recon)
        self.lambda_missing = float(lambda_missing)

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        z0_hat: torch.Tensor,
        z0_target: torch.Tensor,
        mask: torch.Tensor,
        sqrt_alphas_cumprod: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate composite diffusion loss.

        Args:
            eps_pred: Predicted noise tensor [B, T, D].
            eps_target: Ground truth noise tensor [B, T, D].
            z0_hat: Recovered clean latent prediction [B, T, D].
            z0_target: Clean target latent z_0 [B, T, D].
            mask: Observation mask [B, T, 1] (1=observed, 0=missing).
            sqrt_alphas_cumprod: Batch tensor of sqrt(alpha_bar_t) [B, 1, 1].

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # 1. Standard DDPM Noise Prediction Loss
        l_diff = F.mse_loss(eps_pred, eps_target)

        # 2. SNR-weighted x0 Reconstruction Loss (weight by alpha_bar_t to prevent large-t gradient explosion)
        alpha_bar_t = sqrt_alphas_cumprod ** 2
        diff_sq = (z0_hat - z0_target) ** 2
        l_recon = (diff_sq * alpha_bar_t).mean()

        # 3. SNR-weighted Missing-frame Specific Loss
        missing_mask = (1.0 - mask)  # [B, T, 1]
        num_missing = missing_mask.sum() * z0_target.shape[-1]

        if num_missing > 0:
            diff_sq_missing = diff_sq * missing_mask
            l_missing = (diff_sq_missing * alpha_bar_t).sum() / torch.clamp(num_missing, min=1.0)
        else:
            l_missing = torch.tensor(0.0, device=z0_target.device)

        # 4. Composite Loss
        total_loss = (
            (self.lambda_diff * l_diff)
            + (self.lambda_recon * l_recon)
            + (self.lambda_missing * l_missing)
        )

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "diff_loss": float(l_diff.item()),
            "recon_loss": float(l_recon.item()),
            "missing_loss": float(l_missing.item()),
        }

        return total_loss, loss_dict

    @staticmethod
    def reconstruction_metrics(
        z_hat: torch.Tensor,
        z_0: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute full sequence, missing-frame, and observed-frame metrics."""
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)

        diff = z_hat - z_0
        full_mse = float(torch.mean(diff ** 2).item())
        full_mae = float(torch.mean(torch.abs(diff)).item())

        missing_mask = (1.0 - mask)
        num_missing = missing_mask.sum() * z_0.shape[-1]
        if num_missing > 0:
            missing_mse = float((((diff * missing_mask) ** 2).sum() / num_missing).item())
            missing_mae = float(((torch.abs(diff) * missing_mask).sum() / num_missing).item())
        else:
            missing_mse = 0.0
            missing_mae = 0.0

        num_observed = mask.sum() * z_0.shape[-1]
        if num_observed > 0:
            observed_mse = float((((diff * mask) ** 2).sum() / num_observed).item())
            observed_mae = float(((torch.abs(diff) * mask).sum() / num_observed).item())
        else:
            observed_mse = 0.0
            observed_mae = 0.0

        return {
            "full_mse": full_mse,
            "full_mae": full_mae,
            "missing_mse": missing_mse,
            "missing_mae": missing_mae,
            "observed_mse": observed_mse,
            "observed_mae": observed_mae,
        }
