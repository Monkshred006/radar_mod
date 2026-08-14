"""Sanity and Mathematical Integrity Tests for Module 05 Latent Diffusion."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from module_05_latent_diffusion.denoiser import LightweightDenoiser
from module_05_latent_diffusion.scheduler import DDPMScheduler
from module_05_latent_diffusion.corruption import RadarLatentCorruption
from module_05_latent_diffusion.losses import DiffusionLoss
from module_05_latent_diffusion.latent_diffusion import LatentDiffusionModel


class TestDiffusionSanity:
    """Comprehensive sanity tests A through H for latent diffusion."""

    @pytest.fixture
    def scheduler(self):
        return DDPMScheduler(num_train_timesteps=50, beta_schedule="linear")

    @pytest.fixture
    def denoiser(self):
        return LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2)

    def test_h_known_noise_exact_inversion(self, scheduler):
        """TEST H — KNOWN NOISE TEST: Verify recovered z0 equals original z0 when using exact noise."""
        B, T, D = 4, 16, 64
        z0 = torch.randn(B, T, D)
        noise = torch.randn(B, T, D)

        for t_val in [0, 5, 25, 49]:
            t = torch.full((B,), t_val, dtype=torch.long)
            zt = scheduler.add_noise(original_samples=z0, noise=noise, timesteps=t)
            recovered_z0 = scheduler.predict_z0_from_eps(z_t=zt, eps_pred=noise, timesteps=t)

            mse = torch.mean((recovered_z0 - z0) ** 2).item()
            assert mse < 1e-5, f"Known noise test failed at t={t_val}: MSE={mse}"

    def test_g_zero_diffusion_step(self, scheduler):
        """TEST G — ZERO DIFFUSION STEP / t=0: Verify scheduler behavior at t=0."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        noise = torch.zeros(B, T, D)
        t = torch.zeros((B,), dtype=torch.long)

        zt = scheduler.add_noise(original_samples=z0, noise=noise, timesteps=t)
        # At t=0 with 0 noise, z_t should be almost exactly sqrt(alpha_0) * z0 ≈ z0
        assert torch.allclose(zt, z0 * scheduler.sqrt_alphas_cumprod[0], atol=1e-5)

    def test_a_no_corruption_preservation(self, scheduler, denoiser):
        """TEST A — NO CORRUPTION: mask=1 everywhere should preserve input without distortion."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        mask = torch.ones(B, T, 1)

        z_hat = scheduler.reconstruct(
            denoiser=denoiser,
            condition=z0,
            mask=mask,
            num_inference_steps=10,
        )

        # Observed frames must match condition z0 exactly
        obs_mse = torch.mean((z_hat * mask - z0 * mask) ** 2).item()
        assert obs_mse < 1e-6, f"No corruption test failed: obs_mse={obs_mse}"

    def test_b_extreme_corruption(self, scheduler, denoiser):
        """TEST B — 100% / EXTREME CORRUPTION: Model degrades gracefully without NaN/Inf."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        mask = torch.zeros(B, T, 1)  # 100% missing
        zc = z0 * mask

        z_hat = scheduler.reconstruct(
            denoiser=denoiser,
            condition=zc,
            mask=mask,
            num_inference_steps=10,
        )

        assert not torch.isnan(z_hat).any(), "NaN found in extreme corruption output"
        assert not torch.isinf(z_hat).any(), "Inf found in extreme corruption output"
        assert z_hat.shape == (B, T, D)

    def test_c_single_missing_frame(self, scheduler, denoiser):
        """TEST C — SINGLE MISSING FRAME: Exactly one missing frame (index 8)."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        mask = torch.ones(B, T, 1)
        mask[:, 8, :] = 0.0
        zc = z0 * mask

        z_hat = scheduler.reconstruct(
            denoiser=denoiser,
            condition=zc,
            mask=mask,
            num_inference_steps=10,
        )

        # Observed frames preserved
        obs_diff = (z_hat - z0) * mask
        assert torch.mean(obs_diff ** 2).item() < 1e-6

        # Missing frame is populated and finite
        missing_frame = z_hat[:, 8, :]
        assert not torch.isnan(missing_frame).any()
        assert not torch.all(missing_frame == 0.0)

    def test_d_contiguous_temporal_gap(self, scheduler, denoiser):
        """TEST D — CONTIGUOUS TEMPORAL GAP: Frames 6 to 9 missing."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        mask = torch.ones(B, T, 1)
        mask[:, 6:10, :] = 0.0
        zc = z0 * mask

        z_hat = scheduler.reconstruct(
            denoiser=denoiser,
            condition=zc,
            mask=mask,
            num_inference_steps=10,
        )

        # Outside gap is preserved
        obs_mse = torch.mean(((z_hat - z0) * mask) ** 2).item()
        assert obs_mse < 1e-6

        # Inside gap is synthesized
        gap_frames = z_hat[:, 6:10, :]
        assert not torch.isnan(gap_frames).any()

    def test_e_observed_frame_preservation(self, scheduler, denoiser):
        """TEST E — OBSERVED FRAME PRESERVATION: Observed frames have zero drift from z_c."""
        B, T, D = 4, 16, 64
        z0 = torch.randn(B, T, D)
        corruption = RadarLatentCorruption({"enabled": True, "frame_dropout": {"enabled": True, "probability": 0.3}})
        zc, mask = corruption(z0)

        z_hat = scheduler.reconstruct(
            denoiser=denoiser,
            condition=zc,
            mask=mask,
            num_inference_steps=15,
        )

        metrics = DiffusionLoss.reconstruction_metrics(z_hat, z0, mask)
        assert metrics["observed_mse"] < 1e-6, f"Observed MSE drifted: {metrics['observed_mse']}"
