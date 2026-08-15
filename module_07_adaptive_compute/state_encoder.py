"""State encoder for PhotonShield V3 Adaptive Compute Scheduling.

Extracts normalized 9-dimensional state vector from corrupted radar input and initial rapid observable projections.
Does NOT use ground-truth labels, future frames, or clean targets.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_06_physics.radar_constants import DT, MAX_RANGE, MAX_VELOCITY
from module_06_physics.latent_physics_head import LatentPhysicsHead
from module_06_physics.physics_losses import RadarPhysicsLoss


STATE_DIM = 9
STATE_FEATURE_NAMES = [
    "snr_quality",
    "obs_ratio",
    "gap_length",
    "est_range",
    "est_velocity",
    "kin_residual",
    "energy_residual",
    "r_uncertainty",
    "v_uncertainty",
]


class AdaptiveComputeStateEncoder(nn.Module):
    """Extracts normalized 9-dimensional state vector before full diffusion inpainting."""

    def __init__(
        self,
        physics_head: Optional[LatentPhysicsHead] = None,
        dt: float = DT,
    ) -> None:
        super().__init__()
        self.physics_head = physics_head if physics_head is not None else LatentPhysicsHead(latent_dim=64, hidden_dim=32)
        self.dt = float(dt)
        self.raw_extractor = RadarPhysicsLoss(dt=dt, physics_head=self.physics_head).raw_extractor

    def forward(
        self,
        zc: torch.Tensor,
        mask: torch.Tensor,
        x_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract normalized state representation.

        Args:
            zc: Corrupted conditioning latent sequence `[B, T, 64]`.
            mask: Observation mask `[B, T, 1]`, where 1=observed, 0=missing.
            x_raw: Optional raw radar feature sequence `[B, T, 64]` for fast SNR extraction.

        Returns:
            Tuple of (state_tensor `[B, 9]`, state_dict).
        """
        B, T, D = zc.shape
        device = zc.device

        obs = mask[:, :, 0]  # [B, T]
        obs_ratio = torch.mean(obs, dim=1, keepdim=True)  # [B, 1]

        # 1. Compute Mean Gap Length per Sequence
        missing = 1.0 - obs
        # Simple vectorized run length estimation
        left_run = torch.zeros_like(obs)
        for t in range(T):
            if t == 0:
                left_run[:, t] = missing[:, t]
            else:
                left_run[:, t] = (left_run[:, t - 1] + 1.0) * missing[:, t]

        max_gap_per_seq = torch.max(left_run, dim=1, keepdim=True)[0] / float(T)  # [B, 1] normalized

        # 2. Extract initial observables directly from conditioned zc
        with torch.no_grad():
            obs_dict = self.physics_head(zc)
            r_hat = obs_dict["range"]       # [B, T] in meters
            v_hat = obs_dict["velocity"]    # [B, T] in m/s
            e_hat = obs_dict["energy"]      # [B, T]

            # Observables uncertainty (temporal variance over sequence)
            r_var = torch.var(r_hat, dim=1, keepdim=True, unbiased=False)
            v_var = torch.var(v_hat, dim=1, keepdim=True, unbiased=False)
            r_uncertainty = torch.clamp(torch.sqrt(r_var + 1e-8) / MAX_RANGE, 0.0, 1.0)
            v_uncertainty = torch.clamp(torch.sqrt(v_var + 1e-8) / MAX_VELOCITY, 0.0, 1.0)

            # Kinematic residual estimate
            dr_dt = (r_hat[:, 1:] - r_hat[:, :-1]) / self.dt
            v_target = v_hat[:, :-1]
            kin_res = torch.mean(torch.abs(dr_dt - v_target), dim=1, keepdim=True) / MAX_VELOCITY
            kin_res = torch.clamp(kin_res, 0.0, 1.0)

            # Energy continuity
            d_energy = torch.abs(e_hat[:, 1:] - e_hat[:, :-1])
            energy_res = torch.clamp(torch.mean(d_energy, dim=1, keepdim=True), 0.0, 1.0)

            # SNR signal quality
            snr_quality = torch.clamp(torch.mean(e_hat, dim=1, keepdim=True), 0.0, 1.0)

            # Mean physical observables
            est_range = torch.clamp(torch.mean(r_hat, dim=1, keepdim=True) / MAX_RANGE, 0.0, 1.0)
            est_velocity = torch.clamp(torch.mean(torch.abs(v_hat), dim=1, keepdim=True) / MAX_VELOCITY, 0.0, 1.0)

        state_tensor = torch.cat([
            snr_quality,
            obs_ratio,
            max_gap_per_seq,
            est_range,
            est_velocity,
            kin_res,
            energy_res,
            r_uncertainty,
            v_uncertainty,
        ], dim=-1)  # [B, 9]

        state_dict = {
            "snr_quality": snr_quality,
            "obs_ratio": obs_ratio,
            "gap_length": max_gap_per_seq,
            "est_range": est_range,
            "est_velocity": est_velocity,
            "kin_residual": kin_res,
            "energy_residual": energy_res,
            "r_uncertainty": r_uncertainty,
            "v_uncertainty": v_uncertainty,
        }

        return state_tensor, state_dict
