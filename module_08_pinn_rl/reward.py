"""Reward function for Module 8 RL.

Computes:
    R_t = w_task · task_success
        - w_err  · state_error
        - w_phys · physics_violation_penalty   (SEPARATE from PINN L_physics)
        - w_act  · action_cost

IMPORTANT
---------
* Reward weights are DEVELOPMENT PLACEHOLDERS. The final PhotonShield
  reward must be validated against real application objectives.
* physics_violation_penalty is an RL reward-design term, INDEPENDENT of
  PINN L_physics. They are separate quantities for separate optimizers.
* Contribution of physics violation to RL reward is optional and must
  be explicitly configured.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from module_08_pinn_rl.config import EnvironmentConfig, RewardConfig


class RewardFunction:
    """Configurable reward function for the RL agent.

    Parameters
    ----------
    reward_config : RewardConfig
        Weights for each reward component.
    env_config : EnvironmentConfig
        Provides target_position for task success computation.
    """

    def __init__(
        self,
        reward_config: RewardConfig,
        env_config: EnvironmentConfig,
    ) -> None:
        self.cfg = reward_config
        self.target = env_config.target_position

    def compute(
        self,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """Compute scalar reward.

        Parameters
        ----------
        state : ndarray
            State before the action.
        action : int or ndarray
            Action taken.
        next_state : ndarray
            State after the action.
        info : dict
            Environment info dict (may contain 'physics_violation').

        Returns
        -------
        float
            Scalar reward.
        """
        # Task success: proximity to target (position index 0)
        pos = float(next_state[0])
        dist = abs(pos - self.target)
        # Normalise so dist=0 → +1, dist large → 0
        task_success = float(np.exp(-dist))

        # State error: distance from target
        state_error = float(dist)

        # Physics violation (from environment info, if available)
        physics_violation = float(info.get("physics_violation", 0.0))

        # Action cost
        if isinstance(action, int):
            action_cost = float(abs(action))
        else:
            action_cost = float(np.linalg.norm(action))

        reward = (
            self.cfg.weight_task_success * task_success
            - self.cfg.weight_state_error * state_error
            - self.cfg.weight_physics_violation * physics_violation
            - self.cfg.weight_action_cost * action_cost
        )
        return float(reward)
