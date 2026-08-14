"""Tests for strict causality and streaming invariance in Module 8."""

import pytest
import torch

from module_08_pinn_rl.config import DynamicsConfig, RLConfig, RLStateConfig
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.rl_policy import MLPPolicy
from module_08_pinn_rl.state import RLStateBuilder


class TestCausality:
    def test_streaming_state_causality(self):
        builder = RLStateBuilder(RLStateConfig(mamba_latent_dim=64))

        # Timestep 1
        z1 = torch.randn(64)
        s1 = builder.build(z1, 0.8, 0.1, [20.0, 50.0, 1010.0])

        # Timestep 2 (future)
        z2 = torch.randn(64)
        s2 = builder.build(z2, 0.2, 0.9, [30.0, 80.0, 1005.0])

        # State 1 must remain strictly invariant to creation of State 2
        assert torch.equal(s1.components["mamba_latent"], z1)
        assert s1.components["target_probability"].item() == pytest.approx(0.8)

    def test_policy_action_causality(self):
        cfg = RLConfig(action_dim=3)
        policy = MLPPolicy(state_dim=69, config=cfg)
        policy.eval()

        s_t = torch.randn(69)
        with torch.no_grad():
            act1, _, _, _ = policy.get_action_and_value(s_t)

        # Generating future states does not affect action at s_t
        s_future = torch.randn(69)
        with torch.no_grad():
            _ = policy.get_action_and_value(s_future)
            act1_repeat, _, _, _ = policy.get_action_and_value(s_t)

        assert torch.equal(act1, act1_repeat)

    def test_pinn_dynamics_causality(self):
        cfg = DynamicsConfig(state_dim=10, action_dim=2)
        model = PhysicsInformedDynamicsModel(cfg)
        model.eval()

        s_t = torch.randn(1, 10)
        a_t = torch.tensor([[1.0, 0.0]])

        with torch.no_grad():
            pred_t1 = model(s_t, a_t)

        # Modifying future sequence does not change prediction for step t
        s_future = torch.randn(1, 10)
        a_future = torch.tensor([[0.0, 1.0]])
        with torch.no_grad():
            _ = model(s_future, a_future)
            pred_t1_repeat = model(s_t, a_t)

        assert torch.allclose(pred_t1, pred_t1_repeat)
