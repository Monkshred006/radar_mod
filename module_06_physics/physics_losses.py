"""Physics-Informed Losses for Radar Latents and Physical Observables.

Implements differentiable physics regularizers using LatentPhysicsHead:
1. Kinematic Consistency Loss (dR/dt ≈ velocity)
2. Bounded Acceleration Regularizer (soft penalty on excessive da/dt)
3. Energy & SNR Temporal Continuity Loss
4. Supervised Physical Observable Alignment (optional when clean x is available)
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_06_physics.radar_constants import DT
from module_06_physics.latent_physics_head import LatentPhysicsHead
from module_06_physics.observable_extractor import RadarObservableExtractor


class RadarPhysicsLoss(nn.Module):
    """Composite Physics-Informed Regularization Loss for Radar Latent Diffusion.

    Attributes:
        dt: Temporal sampling interval in seconds (default ~0.03333 s for 30 FPS).
        velocity_sign: Radial velocity sign convention (+1 for receding = positive range-rate).
        lambda_kin: Weight for kinematic consistency loss.
        lambda_acc: Weight for acceleration regularizer.
        lambda_energy: Weight for energy temporal continuity.
        lambda_align: Weight for ground-truth physical observable alignment.
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
        lambda_align: float = 0.5,
        a_ref: float = 5.0,
        tau: float = 1.0,
        physics_head: Optional[LatentPhysicsHead] = None,
    ) -> None:
        super().__init__()
        if dt <= 0:
            raise ValueError(f"Temporal interval dt must be strictly positive, got {dt}")
        self.dt = float(dt)
        self.velocity_sign = int(velocity_sign)
        self.lambda_kin = float(lambda_kin)
        self.lambda_acc = float(lambda_acc)
        self.lambda_energy = float(lambda_energy)
        self.lambda_align = float(lambda_align)
        self.a_ref = float(a_ref)
        self.tau = float(tau)

        self.physics_head = physics_head if physics_head is not None else LatentPhysicsHead()
        self.raw_extractor = RadarObservableExtractor(temperature=temperature)

    def compute_kinematic_loss(self, r_hat: torch.Tensor, v_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute range-rate vs. radial velocity kinematic consistency loss."""
        dr_dt = (r_hat[:, 1:] - r_hat[:, :-1]) / self.dt
        v_target = self.velocity_sign * v_hat[:, :-1]
        kin_residual = dr_dt - v_target
        loss = F.smooth_l1_loss(dr_dt, v_target, beta=1.0)
        return loss, kin_residual

    def compute_acceleration_loss(self, v_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute soft bounded acceleration loss."""
        a_t = (v_hat[:, 1:] - v_hat[:, :-1]) / self.dt
        acc_penalty = F.softplus((torch.abs(a_t) - self.a_ref) / self.tau)
        loss = torch.mean(acc_penalty)
        return loss, a_t

    def compute_energy_loss(self, energy: torch.Tensor) -> torch.Tensor:
        """Compute temporal continuity loss for radar reflection energy."""
        d_energy = energy[:, 1:] - energy[:, :-1]
        return F.smooth_l1_loss(d_energy, torch.zeros_like(d_energy), beta=1.0)

    def forward(
        self,
        latent_hat: torch.Tensor,
        x_clean: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate composite physics loss from reconstructed latent sequence.

        Args:
            latent_hat: Reconstructed continuous latent sequence `[B, T, 64]`.
            x_clean: Optional clean radar input features `[B, T, 64]` for physical alignment.
            mask: Optional observation mask `[B, T, 1]`.

        Returns:
            Tuple of (total_physics_loss, component_dict).
        """
        obs = self.physics_head(latent_hat)
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

        l_align = torch.tensor(0.0, device=latent_hat.device)
        if x_clean is not None:
            with torch.no_grad():
                r_gt = self.raw_extractor.extract_range(x_clean[..., 0:30])
                v_gt = self.raw_extractor.extract_velocity(x_clean[..., 30:60])
            l_r = F.smooth_l1_loss(r_hat, r_gt, beta=1.0)
            l_v = F.smooth_l1_loss(v_hat, v_gt, beta=1.0)
            l_align = l_r + l_v
            total_loss = total_loss + self.lambda_align * l_align

        components = {
            "physics_total": total_loss,
            "loss_kinematic": l_kin,
            "loss_acceleration": l_acc,
            "loss_energy": l_energy,
            "loss_alignment": l_align,
            "r_hat": r_hat,
            "v_hat": v_hat,
            "e_hat": e_hat,
            "kin_residual": kin_res,
            "acceleration": acc_t,
        }

        return total_loss, components
