"""Lightweight Differentiable Physics Observable Head for Learned Latents.

Maps mixed temporal latent states z_hat [B, T, 64] into physically calibrated observables:
- Estimated Target Range R_hat [B, T] in meters [0.0, 15.0]
- Estimated Target Radial Velocity v_hat [B, T] in m/s [-8.32, +8.32]
- Estimated Radar Energy E_hat [B, T]

Uses compact multi-layer perceptron with bounded physical activations (Sigmoid / Tanh).
"""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from module_06_physics.radar_constants import (
    MIN_RANGE,
    MAX_RANGE,
    MIN_VELOCITY,
    MAX_VELOCITY,
)


class LatentPhysicsHead(nn.Module):
    """Lightweight neural decoder mapping mixed latent states [B, T, 64] to physical observables.

    Parameter Count: ~5,347 parameters (< 22 KB FP32).
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
        )

        # Specialized lightweight observable heads
        self.range_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Strict bounded range in [0, 1]
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Strict bounded velocity in [-1, 1]
        )
        self.energy_head = nn.Linear(hidden_dim, 1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract continuous physical observables from latent sequence.

        Args:
            z: Latent tensor `[B, T, 64]`.

        Returns:
            Dict containing:
                - 'range': [B, T] in meters [0.0, 15.0]
                - 'velocity': [B, T] in m/s [-8.32, +8.32]
                - 'energy': [B, T] continuous log-energy proxy
        """
        feat = self.trunk(z)  # [B, T, hidden_dim]

        r_norm = self.range_head(feat).squeeze(-1)  # [B, T]
        r_hat = MIN_RANGE + r_norm * (MAX_RANGE - MIN_RANGE)

        v_norm = self.velocity_head(feat).squeeze(-1)  # [B, T]
        v_hat = v_norm * MAX_VELOCITY

        e_hat = self.energy_head(feat).squeeze(-1)  # [B, T]

        return {
            "range": r_hat,
            "velocity": v_hat,
            "energy": e_hat,
        }
