"""Loss functions for Latent Diffusion Training and Reconstruction Evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Calculates noise prediction MSE loss and latent reconstruction error."""

    def __init__(self, loss_type: str = "mse") -> None:
        super().__init__()
        self.loss_type = loss_type

    def forward(self, eps_pred: torch.Tensor, eps_target: torch.Tensor) -> torch.Tensor:
        """Calculate MSE between predicted noise and actual added noise."""
        return F.mse_loss(eps_pred, eps_target)

    @staticmethod
    def reconstruction_mse(z_hat: torch.Tensor, z_0: torch.Tensor) -> float:
        """Compute scalar MSE between reconstructed latent z_hat and clean latent z_0."""
        return float(F.mse_loss(z_hat, z_0).item())
