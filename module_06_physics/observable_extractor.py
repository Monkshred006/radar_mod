"""Differentiable Physical Observable Extractor for Radar Latents.

Maps continuous latent feature vectors [B, T, 64] to physical radar observables:
- Estimated Target Range R_hat [B, T] in meters [0.0, 15.0]
- Estimated Target Radial Velocity v_hat [B, T] in m/s [-8.32, +8.32]
- Radar Energy / Intensity E_hat [B, T]
- Estimated Signal-to-Noise Ratio SNR_hat [B, T]
"""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_06_physics.radar_constants import (
    MIN_RANGE,
    MAX_RANGE,
    NUM_RANGE_FEATS,
    MIN_VELOCITY,
    MAX_VELOCITY,
    NUM_DOPPLER_FEATS,
)


class RadarObservableExtractor(nn.Module):
    """Differentiable SoftArgmax physical observable extractor for radar latents.

    Attributes:
        temperature: Softmax scaling temperature (> 0) controlling peak sharpness.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = float(temperature)

        # Pre-compute fixed 1D coordinate grids
        # Range axis: [0.0, 15.0] meters across 30 feature bins
        range_axis = torch.linspace(MIN_RANGE, MAX_RANGE, NUM_RANGE_FEATS, dtype=torch.float32)
        # Velocity axis: [-8.32, +8.32] m/s across 30 feature bins
        velocity_axis = torch.linspace(MIN_VELOCITY, MAX_VELOCITY, NUM_DOPPLER_FEATS, dtype=torch.float32)

        self.register_buffer("range_axis", range_axis)
        self.register_buffer("velocity_axis", velocity_axis)

    def extract_range(self, range_profile: torch.Tensor) -> torch.Tensor:
        """Extract continuous range estimate in meters via SoftArgmax.

        Args:
            range_profile: Range profile tensor `[..., 30]`.

        Returns:
            Continuous range tensor `[...]` in meters.
        """
        weights = F.softmax(range_profile / self.temperature, dim=-1)
        axis = self.range_axis.to(range_profile.device, dtype=range_profile.dtype)
        return torch.sum(weights * axis, dim=-1)

    def extract_velocity(self, doppler_profile: torch.Tensor) -> torch.Tensor:
        """Extract continuous radial velocity estimate in m/s via SoftArgmax.

        Args:
            doppler_profile: Doppler profile tensor `[..., 30]`.

        Returns:
            Continuous velocity tensor `[...]` in m/s.
        """
        weights = F.softmax(doppler_profile / self.temperature, dim=-1)
        axis = self.velocity_axis.to(doppler_profile.device, dtype=doppler_profile.dtype)
        return torch.sum(weights * axis, dim=-1)

    def extract_energy(self, latent: torch.Tensor) -> torch.Tensor:
        """Extract continuous average radar reflection energy.

        Args:
            latent: Latent tensor `[B, T, 64]`.

        Returns:
            Log-energy tensor `[B, T]`.
        """
        # Latent profile slice [0:60] captures Range + Doppler energy distributions
        profile_feats = latent[..., 0:60]
        # Use log-sum-exp or mean of squares for continuous smooth energy proxy
        energy = torch.mean(profile_feats ** 2, dim=-1) + 1e-6
        return torch.log(energy)

    def extract_snr(self, latent: torch.Tensor) -> torch.Tensor:
        """Extract estimated signal-to-noise ratio.

        Args:
            latent: Latent tensor `[B, T, 64]`.

        Returns:
            SNR proxy tensor `[B, T]`.
        """
        # Summary scalar at index 63 represents normalized SNR proxy
        snr_feat = latent[..., 63]
        return snr_feat

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract all physical observables from a batch of temporal latents.

        Args:
            latent: Latent sequence `[B, T, 64]`.

        Returns:
            Dict containing:
                - 'range': [B, T] in meters
                - 'velocity': [B, T] in m/s
                - 'energy': [B, T] log-energy
                - 'snr': [B, T] SNR proxy
        """
        if latent.shape[-1] < 64:
            raise ValueError(f"Expected last dimension >= 64, got shape {latent.shape}")

        range_prof = latent[..., 0:30]
        doppler_prof = latent[..., 30:60]

        r_hat = self.extract_range(range_prof)
        v_hat = self.extract_velocity(doppler_prof)
        e_hat = self.extract_energy(latent)
        snr_hat = self.extract_snr(latent)

        return {
            "range": r_hat,
            "velocity": v_hat,
            "energy": e_hat,
            "snr": snr_hat,
        }
