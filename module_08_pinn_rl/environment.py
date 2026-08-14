"""RL Environments for Module 8.

SyntheticKinematicEnv — SYNTHETIC VERIFICATION ONLY
    State: [position, velocity]
    Actions: discrete {0: decelerate, 1: maintain, 2: accelerate} or continuous
    Dynamics: x_{t+1} = x_t + v_t*dt + noise
              v_{t+1} = v_t + a*dt + noise
    Goal: reach target position
    Purpose: verify that the PINN + RL pipeline operates correctly.
    Does NOT represent real PhotonShield hardware dynamics.

PhotonShieldRLEnv — stub for future real/replay environment
    Will use Module 4/7 outputs as observations.
    Requires real or replay data; not yet implemented.

Both environments provide a Gymnasium-compatible API:
    reset(seed=None) → (observation, info)
    step(action)     → (observation, reward, terminated, truncated, info)

No gymnasium package is required.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from module_08_pinn_rl.action import SYNTHETIC_KINEMATIC_ACTION_MAP
from module_08_pinn_rl.config import EnvironmentConfig, RewardConfig
from module_08_pinn_rl.reward import RewardFunction


class SyntheticKinematicEnv:
    """2-D kinematic environment — SYNTHETIC VERIFICATION ONLY.

    State: [x, v] where x = position, v = velocity.

    Discrete actions map to acceleration forces:
        0 → a = -1.0   (decelerate)
        1 → a =  0.0   (maintain)
        2 → a = +1.0   (accelerate)

    Continuous action: scalar acceleration directly.

    Dynamics (SYNTHETIC, NOT real PhotonShield hardware):
        x_{t+1} = x_t + v_t * dt + noise
        v_{t+1} = v_t + a_t * dt + noise

    Episode ends when:
        - |x - target| < 0.05 (success), or
        - max_steps reached (truncated), or
        - |x| > 10 (out of bounds, terminated)

    Deterministic with fixed seed (no stochastic sampling is applied
    when noise_std=0).
    """

    ACTION_FORCE_MAP = SYNTHETIC_KINEMATIC_ACTION_MAP  # {0: -1, 1: 0, 2: +1}
    SUCCESS_THRESHOLD = 0.05
    BOUNDARY = 10.0

    def __init__(
        self,
        config: EnvironmentConfig,
        reward_fn: Optional[RewardFunction] = None,
    ) -> None:
        self.config = config
        self.reward_fn = reward_fn or RewardFunction(
            RewardConfig(), config
        )
        self._rng: Optional[np.random.Generator] = None
        self._state = np.zeros(2, dtype=np.float32)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        return (self.config.state_dim,)

    @property
    def action_space_n(self) -> int:
        """Number of discrete actions (discrete mode only)."""
        return self.config.n_discrete_actions

    # ------------------------------------------------------------------
    # Gymnasium-compatible API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            RNG seed for determinism. If None, uses config.seed.

        Returns
        -------
        (observation, info)
        """
        effective_seed = seed if seed is not None else self.config.seed
        self._rng = np.random.default_rng(effective_seed)
        # Random initial position/velocity near origin
        self._state = self._rng.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self._step_count = 0
        return self._state.copy(), {"step": 0}

    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply action and return (obs, reward, terminated, truncated, info).

        Parameters
        ----------
        action : int (discrete) or float (continuous)

        Returns
        -------
        (observation, reward, terminated, truncated, info)
            Gymnasium-compatible 5-tuple.
        """
        if self._rng is None:
            raise RuntimeError("Call reset() before step().")

        prev_state = self._state.copy()

        # Map action to scalar force
        if self.config.action_type == "discrete":
            a = float(self.ACTION_FORCE_MAP[int(action)])
        else:
            a = float(action)

        # Kinematic dynamics with optional noise
        noise = self._rng.normal(0, self.config.noise_std, size=2).astype(np.float32)
        x_new = self._state[0] + self._state[1] * self.config.dt + noise[0]
        v_new = self._state[1] + a * self.config.dt + noise[1]
        self._state = np.array([x_new, v_new], dtype=np.float32)
        self._step_count += 1

        # Reward
        info: Dict[str, Any] = {"step": self._step_count, "action_force": a}
        reward = self.reward_fn.compute(prev_state, action, self._state, info)

        # Termination conditions
        dist = abs(float(self._state[0]) - self.config.target_position)
        out_of_bounds = abs(float(self._state[0])) > self.BOUNDARY
        success = dist < self.SUCCESS_THRESHOLD

        terminated = out_of_bounds
        truncated = (self._step_count >= self.config.max_steps) and not terminated

        info["success"] = success
        info["dist_to_target"] = float(dist)
        info["physics_violation"] = 0.0   # synthetic env, no PINN — violation is 0
        return self._state.copy(), float(reward), terminated, truncated, info

    def seed(self, s: int) -> None:
        self._rng = np.random.default_rng(s)


class PhotonShieldRLEnv:
    """Stub for a future Module 4/7-backed RL environment.

    This environment will use real or replay PhotonShield sensor data
    with Module 4/7 representations as observations. It is NOT yet
    implemented and requires real data and hardware validation.

    IMPORTANT: All results from SyntheticKinematicEnv are labelled as
    SYNTHETIC VERIFICATION and do not transfer to this environment.
    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        raise NotImplementedError(
            "PhotonShieldRLEnv is a placeholder for future real/replay data. "
            "Use SyntheticKinematicEnv for algorithm verification."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        raise NotImplementedError


def build_environment(config: EnvironmentConfig) -> SyntheticKinematicEnv:
    """Factory: create the configured environment."""
    if config.env_type == "synthetic_kinematic":
        return SyntheticKinematicEnv(config)
    elif config.env_type == "photonshield":
        raise NotImplementedError(
            "PhotonShieldRLEnv requires real/replay data and is not yet implemented."
        )
    else:
        raise ValueError(f"Unknown env_type: '{config.env_type}'.")
