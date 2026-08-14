"""Tests for SyntheticKinematicEnv and Environment factories."""

import pytest
import numpy as np

from module_08_pinn_rl.config import EnvironmentConfig
from module_08_pinn_rl.environment import SyntheticKinematicEnv, build_environment


class TestEnvironment:
    def test_reset_shape_and_info(self):
        cfg = EnvironmentConfig(state_dim=2, seed=42)
        env = SyntheticKinematicEnv(cfg)
        obs, info = env.reset()

        assert obs.shape == (2,)
        assert isinstance(info, dict)
        assert "step" in info

    def test_step_contract(self):
        cfg = EnvironmentConfig(state_dim=2, seed=42)
        env = SyntheticKinematicEnv(cfg)
        env.reset()

        obs, reward, term, trunc, info = env.step(1)  # Action 1: maintain
        assert obs.shape == (2,)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

    def test_determinism_with_fixed_seed(self):
        cfg1 = EnvironmentConfig(state_dim=2, seed=123, noise_std=0.0)
        env1 = SyntheticKinematicEnv(cfg1)
        obs1, _ = env1.reset(seed=123)
        obs1_step, r1, _, _, _ = env1.step(2)

        cfg2 = EnvironmentConfig(state_dim=2, seed=123, noise_std=0.0)
        env2 = SyntheticKinematicEnv(cfg2)
        obs2, _ = env2.reset(seed=123)
        obs2_step, r2, _, _, _ = env2.step(2)

        assert np.allclose(obs1, obs2)
        assert np.allclose(obs1_step, obs2_step)
        assert r1 == pytest.approx(r2)

    def test_truncation_on_max_steps(self):
        cfg = EnvironmentConfig(state_dim=2, max_steps=5, seed=42)
        env = SyntheticKinematicEnv(cfg)
        env.reset()

        for _ in range(4):
            _, _, term, trunc, _ = env.step(1)
            assert not trunc

        _, _, term, trunc, _ = env.step(1)
        assert trunc

    def test_factory_build_synthetic(self):
        cfg = EnvironmentConfig(env_type="synthetic_kinematic")
        env = build_environment(cfg)
        assert isinstance(env, SyntheticKinematicEnv)

    def test_factory_build_unsupported_raises(self):
        cfg = EnvironmentConfig(env_type="photonshield")
        with pytest.raises(NotImplementedError):
            build_environment(cfg)
