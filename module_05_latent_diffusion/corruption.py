"""Radar-Specific Latent State Corruption Operators for PhotonShield AI V1.

Provides controlled degradation operators on temporal latent representations [B, T, D]
and returns both the corrupted latent `z_c` and explicit binary observation mask `mask`:
- `mask = 1`: observed / valid frame
- `mask = 0`: corrupted / missing frame
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class RadarLatentCorruption(nn.Module):
    """Applies controlled corruption to latent radar state tensors [B, T, D] and returns (z_c, mask)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

        # 1. Gaussian Noise
        cfg_gn = self.config.get("gaussian_noise", {})
        self.gn_enabled = cfg_gn.get("enabled", False)
        self.gn_sigma = float(cfg_gn.get("sigma", 0.05))

        # 2. Frame Dropout (Primary V1.0)
        cfg_fd = self.config.get("frame_dropout", {})
        self.fd_enabled = cfg_fd.get("enabled", True)
        self.fd_prob = float(cfg_fd.get("probability", 0.20))

        # 3. Temporal Gap
        cfg_tg = self.config.get("temporal_gap", {})
        self.tg_enabled = cfg_tg.get("enabled", False)
        self.tg_length = int(cfg_tg.get("gap_length", 4))

        # 4. Amplitude Scaling
        cfg_as = self.config.get("amplitude_scaling", {})
        self.as_enabled = cfg_as.get("enabled", False)
        self.as_min = float(cfg_as.get("min_scale", 0.5))
        self.as_max = float(cfg_as.get("max_scale", 1.5))

        # 5. Random Frame Masking
        cfg_rm = self.config.get("random_masking", {})
        self.rm_enabled = cfg_rm.get("enabled", False)
        self.rm_ratio = float(cfg_rm.get("mask_ratio", 0.25))

    def apply_frame_dropout(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly drop (zero out) individual frames along sequence length T with probability `p`."""
        B, T, D = z.shape
        if not self.fd_enabled or self.fd_prob <= 0:
            return z, torch.ones(B, T, 1, device=z.device)

        # Mask of shape [B, T, 1] with 1 for observed, 0 for dropped
        mask = (torch.rand(B, T, 1, device=z.device) >= self.fd_prob).float()

        # Ensure at least 1 frame remains uncorrupted to maintain signal anchors
        all_zero = (mask.sum(dim=1, keepdim=True) == 0)
        if all_zero.any():
            for b in range(B):
                if all_zero[b, 0, 0]:
                    mask[b, 0, :] = 1.0

        z_c = z * mask
        return z_c, mask

    def apply_gaussian_noise(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add zero-mean Gaussian noise with standard deviation sigma."""
        B, T, D = z.shape
        if not self.gn_enabled or self.gn_sigma <= 0:
            return z, torch.ones(B, T, 1, device=z.device)
        noise = torch.randn_like(z) * self.gn_sigma
        # Noisy frames are considered partially observed/corrupted (mask=0)
        mask = torch.zeros(B, T, 1, device=z.device)
        return z + noise, mask

    def apply_temporal_gap(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero out a contiguous chunk of length `gap_length`."""
        B, T, D = z.shape
        mask = torch.ones(B, T, 1, device=z.device)
        if not self.tg_enabled or self.tg_length <= 0:
            return z, mask

        z_corrupted = z.clone()
        gap_len = min(self.tg_length, T - 1)

        for b in range(B):
            start = torch.randint(0, T - gap_len + 1, (1,)).item()
            z_corrupted[b, start : start + gap_len, :] = 0.0
            mask[b, start : start + gap_len, :] = 0.0

        return z_corrupted, mask

    def apply_amplitude_scaling(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random scale factor per sequence."""
        B, T, D = z.shape
        if not self.as_enabled:
            return z, torch.ones(B, T, 1, device=z.device)
        scales = (torch.rand(B, 1, 1, device=z.device) * (self.as_max - self.as_min)) + self.as_min
        mask = torch.zeros(B, T, 1, device=z.device)
        return z * scales, mask

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Corrupt input latent tensor z [B, T, D].

        Returns:
            Tuple of:
                - z_c: [B, T, D] corrupted latent tensor
                - mask: [B, T, 1] binary observation mask (1=observed, 0=corrupted)
        """
        B, T, D = z.shape
        if not self.enabled:
            return z, torch.ones(B, T, 1, device=z.device)

        z_c = z.clone()
        mask = torch.ones(B, T, 1, device=z.device)

        # Apply active corruptions sequentially, combining masks
        if self.fd_enabled:
            z_c, m_fd = self.apply_frame_dropout(z_c)
            mask = mask * m_fd

        if self.tg_enabled:
            z_c, m_tg = self.apply_temporal_gap(z_c)
            mask = mask * m_tg

        if self.gn_enabled:
            z_c, m_gn = self.apply_gaussian_noise(z_c)
            mask = mask * m_gn

        if self.as_enabled:
            z_c, m_as = self.apply_amplitude_scaling(z_c)
            mask = mask * m_as

        return z_c, mask
