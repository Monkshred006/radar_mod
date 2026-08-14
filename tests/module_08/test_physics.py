"""Tests for PhysicsModel implementations and factory."""

import pytest
import torch

from module_08_pinn_rl.config import PhysicsConfig
from module_08_pinn_rl.physics import (
    KinematicPhysicsModel,
    NoPhysicsModel,
    WaveConvectionPhysicsModel,
    build_physics_model,
)


class TestPhysicsModels:
    def test_kinematic_zero_residual_on_exact_transition(self):
        cfg = PhysicsConfig(physics_model="kinematic", dt=0.1)
        model = KinematicPhysicsModel(cfg)

        # State: x=1.0, v=2.0
        # Action: force = 3.0 (continuous / dim 0)
        # Expected next: x_next = 1.0 + 2.0*0.1 = 1.2, v_next = 2.0 + 3.0*0.1 = 2.3
        state = torch.tensor([[1.0, 2.0]])
        action = torch.tensor([[3.0]])
        exact_next = torch.tensor([[1.2, 2.3]])

        res = model.residual(state, action, exact_next)
        assert torch.allclose(res, torch.zeros_like(res), atol=1e-6)

    def test_kinematic_nonzero_residual_on_inconsistent_transition(self):
        cfg = PhysicsConfig(physics_model="kinematic", dt=0.1)
        model = KinematicPhysicsModel(cfg)

        state = torch.tensor([[1.0, 2.0]])
        action = torch.tensor([[3.0]])
        wrong_next = torch.tensor([[5.0, 10.0]])  # Inconsistent

        res = model.residual(state, action, wrong_next)
        assert not torch.allclose(res, torch.zeros_like(res))

    def test_no_physics_model_returns_zero(self):
        model = NoPhysicsModel()
        s = torch.randn(4, 10)
        a = torch.randn(4, 2)
        next_s = torch.randn(4, 10)

        res = model.residual(s, a, next_s)
        assert res.item() == 0.0

    def test_wave_convection_autograd_residual(self):
        cfg = PhysicsConfig(physics_model="wave_convection", wave_velocity=1.0, wave_diffusion=0.01)
        model = WaveConvectionPhysicsModel(cfg)

        # Field function: u(x, t) = sin(x - t)
        def u_fn(x, t):
            return torch.sin(x - t)

        x = torch.linspace(0, 1, 10).unsqueeze(-1)
        t = torch.linspace(0, 1, 10).unsqueeze(-1)

        res = model.compute_pde_residual(u_fn, x, t)
        assert res.shape == (10, 1)
        assert torch.isfinite(res).all()

    def test_factory_builds_correct_models(self):
        m_kin = build_physics_model(PhysicsConfig(physics_model="kinematic"))
        assert isinstance(m_kin, KinematicPhysicsModel)

        m_wave = build_physics_model(PhysicsConfig(physics_model="wave_convection"))
        assert isinstance(m_wave, WaveConvectionPhysicsModel)

        m_none = build_physics_model(PhysicsConfig(physics_model="none"))
        assert isinstance(m_none, NoPhysicsModel)
