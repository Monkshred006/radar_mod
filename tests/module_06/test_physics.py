"""Unit tests for PhotonShield AI Phase V2 Physics Module (module_06_physics)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from module_06_physics.radar_constants import (
    C,
    FC,
    WAVELENGTH,
    CHIRP_SLOPE,
    MIN_RANGE,
    MAX_RANGE,
    MIN_VELOCITY,
    MAX_VELOCITY,
    DT,
)
from module_06_physics.observable_extractor import RadarObservableExtractor
from module_06_physics.fmcw_model import (
    velocity_to_doppler_shift,
    doppler_shift_to_velocity,
    range_to_beat_frequency,
    beat_frequency_to_range,
    normalized_to_physical_range,
    normalized_to_physical_velocity,
)
from module_06_physics.physics_losses import RadarPhysicsLoss
from module_06_physics.diagnostics import PhysicsDiagnostics


class TestPhysicsModule:
    """Test suite covering physical constants, differentiable extractors, and physics losses."""

    def test_1_fmcw_equations_consistency(self):
        """Verify mathematical inversion consistency of FMCW radar equations."""
        v_test = torch.tensor([0.0, 1.5, -3.2, 5.0, -8.32], dtype=torch.float32)
        f_d = velocity_to_doppler_shift(v_test)
        v_rec = doppler_shift_to_velocity(f_d)
        assert torch.allclose(v_test, v_rec, atol=1e-5), "Velocity <-> Doppler inversion failed"

        r_test = torch.tensor([0.5, 3.0, 7.5, 12.0, 15.0], dtype=torch.float32)
        f_b = range_to_beat_frequency(r_test)
        r_rec = beat_frequency_to_range(f_b)
        assert torch.allclose(r_test, r_rec, atol=1e-5), "Range <-> Beat frequency inversion failed"

    def test_2_range_observable_extractor(self):
        """Verify range observable extraction on synthetic peaked profile."""
        extractor = RadarObservableExtractor(temperature=0.01)
        B, T = 2, 4
        # Create profile peaked at bin index 10 (out of 30 bins: R = 10 * 15 / 29 ≈ 5.17 m)
        target_bin = 10
        expected_r = float(MIN_RANGE + target_bin * (MAX_RANGE - MIN_RANGE) / 29.0)

        profile = torch.full((B, T, 30), -10.0)
        profile[:, :, target_bin] = 10.0  # sharp peak

        r_hat = extractor.extract_range(profile)
        assert r_hat.shape == (B, T)
        assert torch.allclose(r_hat, torch.tensor(expected_r), atol=0.05), (
            f"Extracted range {r_hat[0, 0]} does not match expected {expected_r}"
        )

    def test_3_velocity_observable_extractor(self):
        """Verify velocity observable extraction on synthetic peaked profile."""
        extractor = RadarObservableExtractor(temperature=0.01)
        B, T = 2, 4
        # Create profile peaked at bin index 18 (out of 30 bins: v = -8.32 + 18 * 16.64 / 29 ≈ 2.01 m/s)
        target_bin = 18
        expected_v = float(MIN_VELOCITY + target_bin * (MAX_VELOCITY - MIN_VELOCITY) / 29.0)

        profile = torch.full((B, T, 30), -10.0)
        profile[:, :, target_bin] = 10.0

        v_hat = extractor.extract_velocity(profile)
        assert v_hat.shape == (B, T)
        assert torch.allclose(v_hat, torch.tensor(expected_v), atol=0.05), (
            f"Extracted velocity {v_hat[0, 0]} does not match expected {expected_v}"
        )

    def test_4_kinematic_loss_exact_motion(self):
        """Verify kinematic loss is zero for a trajectory obeying dR/dt = v."""
        physics_loss = RadarPhysicsLoss(dt=DT, velocity_sign=1)
        B, T = 2, 8
        v_const = 3.0  # m/s
        # R(t) = R_0 + v * t * dt
        r_trajectory = 2.0 + v_const * torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, T) * DT
        v_trajectory = torch.full((B, T), v_const, dtype=torch.float32)

        loss, kin_res = physics_loss.compute_kinematic_loss(r_trajectory, v_trajectory)
        assert loss.item() < 1e-4, f"Kinematic loss {loss.item()} should be ~0.0 for exact motion"
        assert torch.allclose(kin_res, torch.zeros_like(kin_res), atol=1e-4)

    def test_5_acceleration_loss_constant_velocity(self):
        """Verify acceleration loss is near minimum for constant velocity."""
        physics_loss = RadarPhysicsLoss(dt=DT, a_ref=5.0, tau=1.0)
        B, T = 2, 8
        v_trajectory = torch.full((B, T), 4.0, dtype=torch.float32)

        loss, acc = physics_loss.compute_acceleration_loss(v_trajectory)
        expected_min = float(torch.nn.functional.softplus(torch.tensor(-5.0)).item())
        assert abs(loss.item() - expected_min) < 1e-4, "Acceleration loss not at minimum for constant velocity"

    def test_6_energy_loss_constant_energy(self):
        """Verify energy continuity loss is zero for steady-state reflection."""
        physics_loss = RadarPhysicsLoss()
        B, T = 2, 8
        energy = torch.full((B, T), -2.5, dtype=torch.float32)
        loss = physics_loss.compute_energy_loss(energy)
        assert loss.item() == 0.0, "Energy continuity loss must be 0.0 for constant energy"

    def test_7_physics_loss_gradient_flow(self):
        """Verify gradients flow cleanly back to latent tensor without NaN/Inf."""
        physics_loss = RadarPhysicsLoss(dt=DT)
        B, T = 4, 16
        z_latent = torch.randn(B, T, 64, requires_grad=True)

        loss, components = physics_loss(z_latent)
        assert not torch.isnan(loss), "Physics loss produced NaN"
        assert not torch.isinf(loss), "Physics loss produced Inf"

        loss.backward()
        assert z_latent.grad is not None, "Gradient did not flow back to latent tensor"
        assert not torch.isnan(z_latent.grad).any(), "Gradient contains NaN"
        assert not torch.isinf(z_latent.grad).any(), "Gradient contains Inf"
        assert z_latent.grad.norm().item() > 0.0, "Gradient norm is zero"

    def test_8_physics_diagnostics(self):
        """Verify diagnostics evaluator reports complete metrics without exception."""
        diag = PhysicsDiagnostics()
        B, T = 4, 16
        z_latent = torch.randn(B, T, 64)

        report = diag.evaluate(z_latent)
        assert "physics_loss" in report
        assert "range_mean" in report
        assert "velocity_mean" in report
        assert report["has_nan"] is False
        assert report["has_inf"] is False
        assert report["out_of_range_range"] is False
        assert report["out_of_range_velocity"] is False

    def test_9_deterministic_physics_calculation(self):
        """Verify physics calculation is strictly deterministic."""
        physics_loss = RadarPhysicsLoss(dt=DT)
        z_latent = torch.randn(4, 16, 64)

        loss1, comp1 = physics_loss(z_latent)
        loss2, comp2 = physics_loss(z_latent)

        diff = float(abs(loss1.item() - loss2.item()))
        assert diff == 0.0, f"Non-deterministic physics loss output: diff = {diff}"
