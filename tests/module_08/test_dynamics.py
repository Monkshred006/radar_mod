"""Tests for PhysicsInformedDynamicsModel."""

import numpy as np
import pytest
import torch

from module_08_pinn_rl.config import DynamicsConfig
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel


class TestDynamicsModel:
    def test_forward_output_shape(self):
        cfg = DynamicsConfig(state_dim=133, action_type="discrete", action_dim=4)
        model = PhysicsInformedDynamicsModel(cfg)

        state = torch.randn(8, 133)
        action = torch.zeros(8, 4)
        action[:, 0] = 1.0

        out = model(state, action)
        assert out.shape == (8, 133)

    def test_backward_gradient_flow(self):
        cfg = DynamicsConfig(state_dim=64, action_type="discrete", action_dim=3)
        model = PhysicsInformedDynamicsModel(cfg)

        state = torch.randn(4, 64)
        action = torch.zeros(4, 3)
        action[:, 1] = 1.0

        out = model(state, action)
        loss = out.pow(2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_predict_numpy_interface(self):
        cfg = DynamicsConfig(state_dim=2, action_type="discrete", action_dim=3)
        model = PhysicsInformedDynamicsModel(cfg)

        s = np.array([0.5, -0.1], dtype=np.float32)
        next_s = model.predict(s, 2)

        assert isinstance(next_s, np.ndarray)
        assert next_s.shape == (2,)

    def test_count_parameters(self):
        cfg = DynamicsConfig(state_dim=10, hidden_dims=[32, 32], action_dim=2)
        model = PhysicsInformedDynamicsModel(cfg)
        count = model.count_parameters()
        assert count > 0
