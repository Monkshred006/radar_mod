"""Unit tests for Diffusion Auxiliary Branch and Noise Scheduler (Phase V1 Preparation)."""

import pytest
import torch
import torch.nn as nn

from module_05_training.noise_scheduler import NoiseScheduler
from module_05_training.diffusion_auxiliary import (
    SinusoidalPosEmb,
    Tiny1DUNetDenoiser,
    DiffusionAuxiliary,
)


class TestDiffusionAuxiliary:
    """Test suite for diffusion modules."""

    def test_noise_scheduler(self):
        scheduler = NoiseScheduler(num_timesteps=10, schedule="linear")
        x_0 = torch.randn(4, 16, 64)
        t = torch.tensor([0, 3, 6, 9])
        x_t, noise = scheduler.q_sample(x_0, t)

        assert x_t.shape == (4, 16, 64)
        assert noise.shape == (4, 16, 64)

    def test_sinusoidal_pos_emb(self):
        emb = SinusoidalPosEmb(dim=64)
        t = torch.tensor([0, 5, 9])
        out = emb(t)
        assert out.shape == (3, 64)

    def test_tiny_1d_unet_denoiser(self):
        denoiser = Tiny1DUNetDenoiser(hidden_dim=64, time_emb_dim=64)
        x_t = torch.randn(2, 16, 64)
        t = torch.tensor([2, 7])
        reconstructed = denoiser(x_t, t)
        assert reconstructed.shape == (2, 16, 64)
        assert not torch.isnan(reconstructed).any()

    def test_diffusion_auxiliary_loss(self):
        latent = torch.randn(4, 16, 64)

        # Disabled by default
        diff_disabled = DiffusionAuxiliary(hidden_dim=64, enabled=False)
        loss_zero = diff_disabled.compute_loss(latent)
        assert loss_zero.item() == 0.0

        # Enabled
        diff_enabled = DiffusionAuxiliary(hidden_dim=64, enabled=True, timesteps=10)
        loss = diff_enabled.compute_loss(latent)
        assert loss.item() > 0.0
        assert not torch.isnan(loss)
