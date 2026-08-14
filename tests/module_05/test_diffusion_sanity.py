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
    """Comprehensive sanity tests A through I for latent diffusion."""

    @pytest.fixture
    def scheduler(self):
        return DDPMScheduler(num_train_timesteps=50, beta_schedule="linear")

    @pytest.fixture
    def denoiser(self):
        return LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2)

    def test_a_scheduler_forward_diffusion(self, scheduler):
        """TEST A — SCHEDULER FORWARD DIFFUSION: Verify q(z_t | z_0) math."""
        B, T, D = 4, 16, 64
        z0 = torch.randn(B, T, D)
        noise = torch.randn(B, T, D)
        t = torch.tensor([0, 10, 25, 49], dtype=torch.long)

        zt = scheduler.add_noise(original_samples=z0, noise=noise, timesteps=t)
        
        # Verify manual formulation
        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        expected = sqrt_alpha * z0 + sqrt_one_minus * noise
        assert torch.allclose(zt, expected, atol=1e-6)

    def test_b_x0_reconstruction(self, scheduler):
        """TEST B — X0 RECONSTRUCTION: Verify clean latent inversion from predicted noise."""
        B, T, D = 4, 16, 64
        z0 = torch.randn(B, T, D)
        noise = torch.randn(B, T, D)
        t = torch.tensor([0, 5, 20, 45], dtype=torch.long)

        zt = scheduler.add_noise(z0, noise, t)
        z0_hat = scheduler.predict_z0_from_eps(z_t=zt, eps_pred=noise, timesteps=t)
        assert torch.allclose(z0_hat, z0, atol=1e-5)

    def test_c_conditional_reconstruction(self, scheduler, denoiser):
        """TEST C — CONDITIONAL RECONSTRUCTION: Reconstruct with conditional mask."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        mask = torch.ones(B, T, 1)
        mask[:, 8, :] = 0.0
        zc = z0 * mask

        z_hat = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=10, deterministic=True)
        assert z_hat.shape == (B, T, D)
        assert not torch.isnan(z_hat).any()
        assert not torch.isinf(z_hat).any()

    def test_d_no_corruption(self, scheduler, denoiser):
        """TEST D — NO CORRUPTION: mask=1 everywhere preserves input with 0.0 error."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        mask = torch.ones(B, T, 1)

        z_hat = scheduler.reconstruct(denoiser, z0, mask, num_inference_steps=10, deterministic=True)
        obs_mse = torch.mean((z_hat - z0) ** 2).item()
        assert obs_mse < 1e-6, f"No corruption test failed: obs_mse={obs_mse}"

    def test_e_observed_frame_preservation(self, scheduler, denoiser):
        """TEST E — OBSERVED FRAME PRESERVATION: Observed frames have zero drift."""
        B, T, D = 4, 16, 64
        z0 = torch.randn(B, T, D)
        corruption = RadarLatentCorruption({"enabled": True, "frame_dropout": {"enabled": True, "probability": 0.3}})
        zc, mask = corruption(z0)

        z_hat = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=15, deterministic=True)
        metrics = DiffusionLoss.reconstruction_metrics(z_hat, z0, mask)
        assert metrics["observed_mse"] < 1e-6, f"Observed MSE drifted: {metrics['observed_mse']}"

    def test_f_missing_frame_reconstruction(self, scheduler, denoiser):
        """TEST F — MISSING FRAME RECONSTRUCTION: Missing frames are synthesized and finite."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        mask = torch.ones(B, T, 1)
        mask[:, 6:10, :] = 0.0  # frames 6-9 missing
        zc = z0 * mask

        z_hat = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=10, deterministic=True)
        missing_frames = z_hat[:, 6:10, :]
        assert not torch.isnan(missing_frames).any()
        assert not torch.all(missing_frames == 0.0)

    def test_g_known_noise_mathematical_inversion(self, scheduler):
        """TEST G — KNOWN NOISE MATHEMATICAL INVERSION: Invert at arbitrary timesteps."""
        B, T, D = 4, 16, 64
        z0 = torch.randn(B, T, D)
        noise = torch.randn(B, T, D)

        for t_val in [0, 5, 25, 49]:
            t = torch.full((B,), t_val, dtype=torch.long)
            zt = scheduler.add_noise(z0, noise, t)
            recovered = scheduler.predict_z0_from_eps(zt, noise, t)
            mse = torch.mean((recovered - z0) ** 2).item()
            assert mse < 1e-5, f"Inversion failed at t={t_val}: MSE={mse}"

    def test_h_latent_consistency(self, scheduler, denoiser):
        """TEST H — LATENT CONSISTENCY & EXTREME CORRUPTION: 100% missing stability."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        mask = torch.zeros(B, T, 1)
        zc = z0 * mask

        z_hat = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=10, deterministic=True)
        assert not torch.isnan(z_hat).any()
        assert not torch.isinf(z_hat).any()
        assert z_hat.shape == (B, T, D)

    def test_i_deterministic_sampling_reproducibility(self, scheduler, denoiser):
        """TEST I — DETERMINISTIC SAMPLING REPRODUCIBILITY: Exactly identical outputs."""
        B, T, D = 2, 16, 64
        z0 = torch.randn(B, T, D)
        mask = torch.ones(B, T, 1)
        mask[:, 4:8, :] = 0.0
        zc = z0 * mask

        out1_det = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=10, deterministic=True)
        out2_det = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=10, deterministic=True)

        max_diff = torch.max(torch.abs(out1_det - out2_det)).item()
        assert max_diff < 1e-7, f"Deterministic mode produced difference: {max_diff}"

        # Stochastic mode variation
        out1_stoch = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=10, deterministic=False)
        out2_stoch = scheduler.reconstruct(denoiser, zc, mask, num_inference_steps=10, deterministic=False)
        diff_stoch = torch.max(torch.abs(out1_stoch - out2_stoch)).item()
        assert diff_stoch > 0.0, "Stochastic mode did not vary"
