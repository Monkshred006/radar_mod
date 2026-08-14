"""Radar-Specific Latent State Corruption Operators for PhotonShield AI V1.

Provides controlled degradation operators on temporal latent representations [B, T, D]:
1. Gaussian Noise
2. Temporal Frame Dropout (Primary for V1.0)
3. Temporal Gaps (Contiguous missing sequence chunks)
4. Amplitude Scaling (Intermittent attenuation)
5. Random Frame Masking
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class RadarLatentCorruption(nn.Module):
    """Applies controlled corruption to latent radar state tensors [B, T, D]."""

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

    def apply_frame_dropout(self, z: torch.Tensor) -> torch.Tensor:
        """Randomly drop (zero out) individual frames along sequence length T with probability `p`."""
        if not self.fd_enabled or self.fd_prob <= 0:
            return z

        B, T, D = z.shape
        # Create mask of shape [B, T, 1]
        mask = (torch.rand(B, T, 1, device=z.device) >= self.fd_prob).float()
        
        # Ensure at least 1 frame remains uncorrupted to maintain signal anchors
        all_zero = (mask.sum(dim=1, keepdim=True) == 0)
        if all_zero.any():
            mask[:, 0, :] = 1.0

        return z * mask

    def apply_gaussian_noise(self, z: torch.Tensor) -> torch.Tensor:
        """Add zero-mean Gaussian noise with standard deviation sigma."""
        if not self.gn_enabled or self.gn_sigma <= 0:
            return z
        noise = torch.randn_like(z) * self.gn_sigma
        return z + noise

    def apply_temporal_gap(self, z: torch.Tensor) -> torch.Tensor:
        """Zero out a contiguous chunk of length `gap_length`."""
        if not self.tg_enabled or self.tg_length <= 0:
            return z

        B, T, D = z.shape
        z_corrupted = z.clone()
        gap_len = min(self.tg_length, T - 1)
        
        for b in range(B):
            start = torch.randint(0, T - gap_len + 1, (1,)).item()
            z_corrupted[b, start : start + gap_len, :] = 0.0
            
        return z_corrupted

    def apply_amplitude_scaling(self, z: torch.Tensor) -> torch.Tensor:
        """Apply random scale factor per sequence."""
        if not self.as_enabled:
            return z
        B, T, D = z.shape
        scales = (torch.rand(B, 1, 1, device=z.device) * (self.as_max - self.as_min)) + self.as_min
        return z * scales

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Corrupt input latent tensor z [B, T, D]."""
        if not self.enabled:
            return z

        z_c = z.clone()

        # Apply active corruptions
        if self.fd_enabled:
            z_c = self.apply_frame_dropout(z_c)
        if self.gn_enabled:
            z_c = self.apply_gaussian_noise(z_c)
        if self.tg_enabled:
            z_c = self.apply_temporal_gap(z_c)
        if self.as_enabled:
            z_c = self.apply_amplitude_scaling(z_c)

        return z_c
