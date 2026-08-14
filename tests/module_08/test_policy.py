"""Tests for MLPPolicy."""

import pytest
import torch

from module_08_pinn_rl.config import RLConfig
from module_08_pinn_rl.rl_policy import MLPPolicy


class TestPolicy:
    def test_discrete_policy_action_sampling(self):
        cfg = RLConfig(action_type="discrete", action_dim=4, hidden_dims=[32, 32])
        policy = MLPPolicy(state_dim=10, config=cfg)

        state = torch.randn(8, 10)
        action, log_prob, entropy, value = policy.get_action_and_value(state)

        assert action.shape == (8,)
        assert (action >= 0).all() and (action < 4).all()
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)
        assert value.shape == (8,)

    def test_continuous_policy_action_sampling(self):
        cfg = RLConfig(action_type="continuous", action_dim=2, hidden_dims=[32, 32])
        policy = MLPPolicy(state_dim=10, config=cfg)

        state = torch.randn(8, 10)
        action, log_prob, entropy, value = policy.get_action_and_value(state)

        assert action.shape == (8, 2)
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)
        assert value.shape == (8,)

    def test_greedy_act(self):
        cfg = RLConfig(action_type="discrete", action_dim=3)
        policy = MLPPolicy(state_dim=4, config=cfg)

        state = torch.randn(4)
        act = policy.act(state)
        assert isinstance(act, int)
        assert 0 <= act < 3

    def test_policy_gradient_flow(self):
        cfg = RLConfig(action_type="discrete", action_dim=3)
        policy = MLPPolicy(state_dim=4, config=cfg)

        state = torch.randn(4, 4)
        _, log_prob, entropy, value = policy.get_action_and_value(state)
        loss = -log_prob.mean() + value.pow(2).mean() - 0.01 * entropy.mean()
        loss.backward()

        for name, param in policy.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
