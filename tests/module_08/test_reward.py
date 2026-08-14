"""Tests for RewardFunction."""

import numpy as np
import pytest

from module_08_pinn_rl.config import EnvironmentConfig, RewardConfig
from module_08_pinn_rl.reward import RewardFunction


class TestRewardFunction:
    def test_proximity_increases_reward(self):
        r_cfg = RewardConfig(weight_task_success=1.0, weight_state_error=0.1)
        env_cfg = EnvironmentConfig(target_position=1.0)
        rf = RewardFunction(r_cfg, env_cfg)

        state = np.array([0.0, 0.0], dtype=np.float32)
        close_state = np.array([0.9, 0.0], dtype=np.float32)
        far_state = np.array([0.1, 0.0], dtype=np.float32)

        r_close = rf.compute(state, 1, close_state, {})
        r_far = rf.compute(state, 1, far_state, {})

        assert r_close > r_far

    def test_physics_violation_penalty(self):
        r_cfg = RewardConfig(weight_physics_violation=0.5)
        env_cfg = EnvironmentConfig(target_position=1.0)
        rf = RewardFunction(r_cfg, env_cfg)

        state = np.array([0.5, 0.0], dtype=np.float32)
        next_state = np.array([0.6, 0.0], dtype=np.float32)

        r_no_viol = rf.compute(state, 1, next_state, {"physics_violation": 0.0})
        r_with_viol = rf.compute(state, 1, next_state, {"physics_violation": 2.0})

        assert r_no_viol - r_with_viol == pytest.approx(0.5 * 2.0)

    def test_action_cost_penalty(self):
        r_cfg = RewardConfig(weight_action_cost=0.1)
        env_cfg = EnvironmentConfig(target_position=1.0)
        rf = RewardFunction(r_cfg, env_cfg)

        state = np.array([0.5, 0.0], dtype=np.float32)
        next_state = np.array([0.6, 0.0], dtype=np.float32)

        r_act0 = rf.compute(state, 0, next_state, {})
        r_act2 = rf.compute(state, 2, next_state, {})

        # Action 2 costs 2 * 0.1 more than action 0
        assert r_act0 > r_act2
