"""State extraction for Adaptive Physics Controller in PhotonShield V3.0."""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_06_physics.radar_constants import MAX_RANGE, MAX_VELOCITY, DT
from module_06_physics.latent_physics_head import LatentPhysicsHead
from module_06_physics.physics_losses import RadarPhysicsLoss


class AdaptivePhysicsStateExtractor:
    """Extracts normalized 10-dimensional state vector from latent reconstruction and observation mask."""

    def __init__(
        self,
        physics_head: LatentPhysicsHead,
        physics_loss_module: RadarPhysicsLoss,
        max_range: float = MAX_RANGE,
        max_velocity: float = MAX_VELOCITY,
    ) -> None:
        self.physics_head = physics_head
        self.physics_loss_module = physics_loss_module
        self.max_range = float(max_range)
        self.max_velocity = float(max_velocity)

    @torch.no_grad()
    def extract_sequence_state(
        self,
        latent_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Extract state vector for a sequence [B, T, D] with mask [B, T, 1].

        Returns:
            Dict containing:
                - state_vector: numpy array [10] or [B, 10]
                - state_dict: dict of individual named features
        """
        B, T, D = latent_hat.shape
        device = latent_hat.device

        obs = mask[:, :, 0]  # [B, T]
        missing = 1.0 - obs

        # 1. Observation ratio [B]
        obs_ratio = torch.mean(obs, dim=1)

        # 2. Current / Mean gap length [B]
        gap_lengths = []
        for b in range(B):
            gaps = []
            curr_gap = 0
            for t in range(T):
                if obs[b, t] < 0.5:
                    curr_gap += 1
                else:
                    if curr_gap > 0:
                        gaps.append(curr_gap)
                    curr_gap = 0
            if curr_gap > 0:
                gaps.append(curr_gap)
            mean_gap = np.mean(gaps) if len(gaps) > 0 else 0.0
            gap_lengths.append(mean_gap / float(T))
        gap_len_tensor = torch.tensor(gap_lengths, device=device, dtype=torch.float32)

        # 3. Observables from LatentPhysicsHead (no ground truth used)
        obs_pred = self.physics_head(latent_hat)
        r_hat = obs_pred["range"]       # [B, T] in meters
        v_hat = obs_pred["velocity"]    # [B, T] in m/s
        e_hat = obs_pred["energy"]      # [B, T] normalized energy

        # 4. Range and Velocity Uncertainty Proxies (temporal variance across sequence)
        r_var = torch.var(r_hat, dim=1, unbiased=False)
        v_var = torch.var(v_hat, dim=1, unbiased=False)
        r_uncertainty = torch.clamp(torch.sqrt(r_var + 1e-8) / self.max_range, 0.0, 1.0)
        v_uncertainty = torch.clamp(torch.sqrt(v_var + 1e-8) / self.max_velocity, 0.0, 1.0)

        # 5. Physical Residuals
        _, comp = self.physics_loss_module(latent_hat)
        kin_res = torch.mean(torch.abs(comp["kin_residual"]), dim=1) / self.max_velocity
        acc_res = torch.mean(torch.abs(comp["acceleration"]), dim=1) / 20.0
        energy_res = torch.mean(torch.abs(e_hat[:, 1:] - e_hat[:, :-1]), dim=1)

        # 6. SNR / Energy Quality
        snr_quality = torch.clamp(torch.mean(e_hat, dim=1), 0.0, 1.0)

        # 7. Normalized Range & Velocity
        est_range = torch.clamp(torch.mean(r_hat, dim=1) / self.max_range, 0.0, 1.0)
        est_velocity = torch.clamp(torch.mean(torch.abs(v_hat), dim=1) / self.max_velocity, 0.0, 1.0)

        # Stack into 10-dimensional state vector: s = [10]
        state_tensor = torch.stack([
            obs_ratio,
            gap_len_tensor,
            r_uncertainty,
            v_uncertainty,
            kin_res,
            acc_res,
            energy_res,
            snr_quality,
            est_range,
            est_velocity,
        ], dim=-1)  # [B, 10]

        return {
            "state_tensor": state_tensor,
            "obs_ratio": obs_ratio.cpu().numpy(),
            "gap_length": gap_len_tensor.cpu().numpy(),
            "r_uncertainty": r_uncertainty.cpu().numpy(),
            "v_uncertainty": v_uncertainty.cpu().numpy(),
            "kin_residual": kin_res.cpu().numpy(),
            "acc_residual": acc_res.cpu().numpy(),
            "energy_residual": energy_res.cpu().numpy(),
            "snr_quality": snr_quality.cpu().numpy(),
            "est_range": est_range.cpu().numpy(),
            "est_velocity": est_velocity.cpu().numpy(),
        }
