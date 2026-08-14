"""Physics-Informed Diagnostics and Integrity Monitoring.

Computes comprehensive physical statistics and validates numerical health:
- Range, velocity, energy distributions
- Kinematic residual and acceleration metrics
- Bounds verification and NaN/Inf detection
"""

from __future__ import annotations

from typing import Dict, Any
import torch

from module_06_physics.radar_constants import (
    MIN_RANGE,
    MAX_RANGE,
    MIN_VELOCITY,
    MAX_VELOCITY,
)
from module_06_physics.physics_losses import RadarPhysicsLoss


class PhysicsDiagnostics:
    """Computes telemetry and health diagnostics for radar physical observables."""

    def __init__(self, physics_loss_module: RadarPhysicsLoss | None = None) -> None:
        self.loss_module = physics_loss_module if physics_loss_module is not None else RadarPhysicsLoss()

    def evaluate(self, latent_hat: torch.Tensor) -> Dict[str, Any]:
        """Compute all physical diagnostic metrics from a reconstructed latent tensor.

        Args:
            latent_hat: Reconstructed latent tensor `[B, T, 64]`.

        Returns:
            Dict containing physical statistics, loss terms, and integrity flags.
        """
        has_nan = bool(torch.isnan(latent_hat).any().item())
        has_inf = bool(torch.isinf(latent_hat).any().item())

        loss_val, components = self.loss_module(latent_hat)

        r_hat = components["r_hat"]
        v_hat = components["v_hat"]
        e_hat = components["e_hat"]
        kin_res = components["kin_residual"]
        acc_t = components["acceleration"]

        out_of_range_r = bool(((r_hat < MIN_RANGE - 1e-3) | (r_hat > MAX_RANGE + 1e-3)).any().item())
        out_of_range_v = bool(((v_hat < MIN_VELOCITY - 1e-3) | (v_hat > MAX_VELOCITY + 1e-3)).any().item())

        return {
            "physics_loss": float(loss_val.item()),
            "loss_kinematic": float(components["loss_kinematic"].item()),
            "loss_acceleration": float(components["loss_acceleration"].item()),
            "loss_energy": float(components["loss_energy"].item()),
            "range_mean": float(torch.mean(r_hat).item()),
            "range_std": float(torch.std(r_hat).item()),
            "range_min": float(torch.min(r_hat).item()),
            "range_max": float(torch.max(r_hat).item()),
            "velocity_mean": float(torch.mean(v_hat).item()),
            "velocity_std": float(torch.std(v_hat).item()),
            "velocity_min": float(torch.min(v_hat).item()),
            "velocity_max": float(torch.max(v_hat).item()),
            "energy_mean": float(torch.mean(e_hat).item()),
            "energy_std": float(torch.std(e_hat).item()),
            "kinematic_residual_mean": float(torch.mean(torch.abs(kin_res)).item()),
            "kinematic_residual_std": float(torch.std(kin_res).item()),
            "acceleration_mean": float(torch.mean(torch.abs(acc_t)).item()),
            "acceleration_std": float(torch.std(acc_t).item()),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "out_of_range_range": out_of_range_r,
            "out_of_range_velocity": out_of_range_v,
        }
