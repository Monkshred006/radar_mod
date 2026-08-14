"""Tests for standalone physics residual computation functions."""

import pytest
import torch

from module_08_pinn_rl.residuals import (
    compute_kinematic_residual,
    compute_wave_convection_residual,
)


class TestResiduals:
    def test_kinematic_residual_zero_for_perfect_dynamics(self):
        # 3 batch items
        state = torch.tensor([[0.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
        action_force = torch.tensor([[0.5], [-0.5], [1.0]])
        dt = 0.1

        # Calculate exact next states:
        # 1: x=0+1*0.1=0.1, v=1+0.5*0.1=1.05
        # 2: x=1-1*0.1=0.9, v=-1-0.5*0.1=-1.05
        # 3: x=2+0*0.1=2.0, v=0+1.0*0.1=0.1
        next_state_exact = torch.tensor([[0.1, 1.05], [0.9, -1.05], [2.0, 0.1]])

        res = compute_kinematic_residual(state, next_state_exact, action_force, dt=dt)
        assert torch.allclose(res, torch.zeros_like(res), atol=1e-6)

    def test_wave_convection_residual_differentiable(self):
        # Parametrized field: u(x, t) = a * sin(x - t)
        a_param = torch.tensor([2.0], requires_grad=True)

        def u_fn(x, t):
            return a_param * torch.sin(x - t)

        x = torch.tensor([[0.5]], requires_grad=True)
        t = torch.tensor([[0.2]], requires_grad=True)

        res = compute_wave_convection_residual(u_fn, x, t, wave_velocity=1.0, wave_diffusion=0.0)
        assert torch.isfinite(res)

        # Loss w.r.t parameter
        loss = res.pow(2).mean()
        loss.backward()

        assert a_param.grad is not None
        assert torch.isfinite(a_param.grad).all()
