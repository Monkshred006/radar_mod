"""Physics-Informed Losses for Radar Latent Sequences.

Implements differentiable physics regularizers:
1. Kinematic Consistency Loss (dR/dt ≈ velocity)
2. Bounded Acceleration Regularizer (soft penalty on excessive da/dt)
3. Energy & SNR Temporal Continuity Loss
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_06_physics.radar_constants import DT
from module_06_physics.observable_extractor import RadarObservableExtractor


class RadarPhysicsLoss(nn.Module):
    """Composite Physics-Informed Regularization Loss for Radar Latent Diffusion.

    Attributes:
        dt: Temporal sampling interval in seconds (default ~0.03333 s for 30 FPS).
        velocity_sign: Radial velocity sign convention (+1 for receding = positive range-rate).
        lambda_kin: Weight for kinematic consistency loss.
        lambda_acc: Weight for acceleration regularizer.
        lambda_energy: Weight for energy temporal continuity.
        a_ref: Reference acceleration threshold in m/s^2 (modeling assumption).
        tau: Softplus temperature for acceleration penalty.
    """

    def __init__(
        self,
        dt: float = DT,
        velocity_sign: int = 1,
        temperature: float = 0.1,
        lambda_kin: float = 1.0,
        lambda_acc: float = 0.1,
        lambda_energy: float = 0.1,
        a_ref: float = 5.0,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        if dt <= 0:
            raise ValueError(f"Temporal interval dt must be strictly positive, got {dt}")
        self.dt = float(dt)
        self.velocity_sign = int(velocity_sign)
        self.lambda_kin = float(lambda_kin)
        self.lambda_acc = float(lambda_acc)
        self.lambda_energy = float(lambda_energy)
        self.a_ref = float(a_ref)
        self.tau = float(tau)

        self.extractor = RadarObservableExtractor(temperature=temperature)

    def compute_kinematic_loss(self, r_hat: torch.Tensor, v_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute range-rate vs. radial velocity kinematic consistency loss.

        Equation:
            dR/dt = (R[t+1] - R[t]) / dt
            residual = dR/dt - velocity_sign * v[t]
            loss = mean(Huber(residual))

        Args:
            r_hat: Estimated range trajectory `[B, T]`.
            v_hat: Estimated velocity trajectory `[B, T]`.

        Returns:
            Tuple of (loss, residual_tensor).
        """
        # Time differences: [B, T-1]
        dr_dt = (r_hat[:, 1:] - r_hat[:, :-1]) / self.dt
        # Midpoint or causal velocity estimate: [B, T-1]
        v_target = self.velocity_sign * v_hat[:, :-1]

        kin_residual = dr_dt - v_target
        loss = F.smooth_l1_loss(dr_dt, v_target, beta=1.0)
        return loss, kin_residual

    def compute_acceleration_loss(self, v_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute soft bounded acceleration loss.

        Equation:
            a[t] = (v[t+1] - v[t]) / dt
            loss = mean(softplus((|a[t]| - a_ref) / tau))

        Args:
            v_hat: Estimated velocity trajectory `[B, T]`.

        Returns:
            Tuple of (loss, acceleration_tensor).
        """
        # Accelerations: [B, T-1]
        a_t = (v_hat[:, 1:] - v_hat[:, :-1]) / self.dt
        # Softplus penalty exceeding a_ref
        acc_penalty = F.softplus((torch.abs(a_t) - self.a_ref) / self.tau)
        loss = torch.mean(acc_penalty)
        return loss, a_t

    def compute_energy_loss(self, energy: torch.Tensor) -> torch.Tensor:
        """Compute temporal continuity loss for radar reflection energy.

        Args:
            energy: Log-energy trajectory `[B, T]`.

        Returns:
            Scalar loss tensor.
        """
        d_energy = energy[:, 1:] - energy[:, :-1]
        return F.smooth_l1_loss(d_energy, torch.zeros_like(d_energy), beta=1.0)

    def forward(
        self,
        latent_hat: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate composite physics loss and component diagnostics.

        Args:
            latent_hat: Reconstructed continuous latent sequence `[B, T, 64]`.
            mask: Optional observation mask `[B, T, 1]`.

        Returns:
            Tuple of (total_physics_loss, component_dict).
        """
        obs = self.extractor(latent_hat)
        r_hat = obs["range"]
        v_hat = obs["velocity"]
        e_hat = obs["energy"]

        l_kin, kin_res = self.compute_kinematic_loss(r_hat, v_hat)
        l_acc, acc_t = self.compute_acceleration_loss(v_hat)
        l_energy = self.compute_energy_loss(e_hat)

        total_loss = (
            self.lambda_kin * l_kin
            + self.lambda_acc * l_acc
            + self.lambda_energy * l_energy
        )

        components = {
            "physics_total": total_loss,
            "loss_kinematic": l_kin,
            "loss_acceleration": l_acc,
            "loss_energy": l_energy,
            "r_hat": r_hat,
            "v_hat": v_hat,
            "e_hat": e_hat,
            "kin_residual": kin_res,
            "acceleration": acc_t,
        }

        return total_loss, components
