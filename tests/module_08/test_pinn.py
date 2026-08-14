"""Tests for PINNLoss computation and backward pass."""

import pytest
import torch

from module_08_pinn_rl.config import DynamicsConfig, PhysicsConfig
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.pinn import PINNLoss


class TestPINNLoss:
    def test_loss_keys_and_scalar(self):
        dyn_cfg = DynamicsConfig(state_dim=2, action_dim=3)
        phys_cfg = PhysicsConfig(physics_model="kinematic", lambda_physics=0.1)
        pinn_loss = PINNLoss(dyn_cfg, phys_cfg)

        s = torch.randn(4, 2)
        a = torch.zeros(4, 3)
        a[:, 0] = 1.0
        pred_next = torch.randn(4, 2)
        true_next = torch.randn(4, 2)

        losses = pinn_loss(pred_next, true_next, s, a)

        assert "loss" in losses
        assert "data_loss" in losses
        assert "physics_loss" in losses
        assert torch.isfinite(losses["loss"])

    def test_lambda_zero_equals_data_loss(self):
        dyn_cfg = DynamicsConfig(state_dim=2, action_dim=3)
        phys_cfg = PhysicsConfig(physics_model="kinematic", lambda_physics=0.0)
        pinn_loss = PINNLoss(dyn_cfg, phys_cfg)

        s = torch.randn(4, 2)
        a = torch.zeros(4, 3)
        a[:, 1] = 1.0
        pred_next = torch.randn(4, 2)
        true_next = torch.randn(4, 2)

        losses = pinn_loss(pred_next, true_next, s, a)
        assert losses["loss"].item() == pytest.approx(losses["data_loss"].item())

    def test_lambda_positive_combines_losses(self):
        dyn_cfg = DynamicsConfig(state_dim=2, action_dim=3)
        phys_cfg = PhysicsConfig(physics_model="kinematic", lambda_physics=0.5)
        pinn_loss = PINNLoss(dyn_cfg, phys_cfg)

        s = torch.randn(4, 2)
        a = torch.zeros(4, 3)
        a[:, 2] = 1.0
        pred_next = torch.randn(4, 2)
        true_next = torch.randn(4, 2)

        losses = pinn_loss(pred_next, true_next, s, a)
        expected = losses["data_loss"].item() + 0.5 * losses["physics_loss"].item()
        assert losses["loss"].item() == pytest.approx(expected, rel=1e-5)

    def test_backward_gradient_flow_to_dynamics_model(self):
        dyn_cfg = DynamicsConfig(state_dim=2, action_dim=3)
        phys_cfg = PhysicsConfig(physics_model="kinematic", lambda_physics=0.1)
        model = PhysicsInformedDynamicsModel(dyn_cfg)
        pinn_loss = PINNLoss(dyn_cfg, phys_cfg)

        s = torch.randn(4, 2)
        a = torch.zeros(4, 3)
        a[:, 0] = 1.0
        true_next = torch.randn(4, 2)

        pred_next = model(s, a)
        losses = pinn_loss(pred_next, true_next, s, a)
        losses["loss"].backward()

        for name, param in model.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
