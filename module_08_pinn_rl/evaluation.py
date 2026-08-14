"""Evaluation utilities for Module 8 PINN + RL.

Provides:
- PINNEvaluator: Evaluates dynamics model (MAE, RMSE, physics residual, constraint violation rate)
- RLEvaluator: Evaluates RL policy in environment (reward, success rate, episode length, action cost)
- ComparisonEvaluator: Evaluates and compares RL-only (Exp A), Data-only Dynamics (Exp B), and RL + PINN (Exp C).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from module_08_pinn_rl.action import ActionEncoder, ActionSpec
from module_08_pinn_rl.config import PINNRLConfig
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.environment import SyntheticKinematicEnv
from module_08_pinn_rl.physics import build_physics_model
from module_08_pinn_rl.rl_policy import MLPPolicy
from module_08_pinn_rl.transitions import Episode


class PINNEvaluator:
    """Evaluates next-state prediction accuracy and physics consistency of a dynamics model."""

    def __init__(
        self,
        dynamics_model: PhysicsInformedDynamicsModel,
        config: PINNRLConfig,
        device: str = "cpu",
    ) -> None:
        self.model = dynamics_model.to(device)
        self.config = config
        self.device = device
        self.physics_model = build_physics_model(config.physics_config)
        self.action_encoder = ActionEncoder(
            ActionSpec(
                action_type=dynamics_model.config.action_type,
                action_dim=dynamics_model.config.action_dim,
            )
        )

    def evaluate(self, episodes: List[Episode]) -> Dict[str, float]:
        """Compute metrics over a collection of test transitions."""
        self.model.eval()
        all_states, all_actions, all_next_states = [], [], []
        for ep in episodes:
            for tr in ep.transitions:
                all_states.append(tr.state.astype(np.float32))
                all_actions.append(self.action_encoder.encode(tr.action).numpy())
                all_next_states.append(tr.next_state.astype(np.float32))

        if not all_states:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "physics_residual": 0.0,
                "constraint_violation_rate": 0.0,
            }

        S = torch.from_numpy(np.array(all_states)).to(self.device)
        A = torch.from_numpy(np.array(all_actions)).to(self.device)
        NS = torch.from_numpy(np.array(all_next_states)).to(self.device)

        with torch.no_grad():
            pred = self.model(S, A)
            diff = pred - NS
            mae = diff.abs().mean().item()
            rmse = (diff.pow(2).mean().sqrt()).item()

            residual = self.physics_model.residual(S, A, pred)
            phys_res = residual.pow(2).mean().item()

            # Constraint violation: residual magnitude above threshold (e.g. 0.1)
            violations = (residual.abs() > 0.1).float().mean().item()

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "physics_residual": float(phys_res),
            "constraint_violation_rate": float(violations),
        }


class RLEvaluator:
    """Evaluates an RL policy inside an environment over multiple test episodes."""

    def __init__(self, config: PINNRLConfig) -> None:
        self.config = config

    def evaluate_policy(
        self,
        policy: MLPPolicy,
        env: SyntheticKinematicEnv,
        n_episodes: int = 10,
        seed: int = 100,
    ) -> Dict[str, float]:
        """Run deterministic evaluation episodes and aggregate performance metrics."""
        policy.eval()
        rewards = []
        lengths = []
        successes = []
        action_costs = []
        latencies = []

        for ep_idx in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep_idx)
            done = False
            total_r = 0.0
            steps = 0
            ep_cost = 0.0

            while not done:
                s_t = torch.from_numpy(obs.astype(np.float32))
                t0 = time.perf_counter()
                action = policy.act(s_t)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

                if isinstance(action, int):
                    ep_cost += abs(action)
                else:
                    ep_cost += float(np.linalg.norm(action))

                next_obs, r, term, trunc, info = env.step(action)
                total_r += r
                steps += 1
                done = term or trunc
                obs = next_obs

                if done:
                    successes.append(1.0 if info.get("success", False) else 0.0)

            rewards.append(total_r)
            lengths.append(steps)
            action_costs.append(ep_cost / max(steps, 1))

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_length": float(np.mean(lengths)),
            "mean_action_cost": float(np.mean(action_costs)),
            "inference_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        }


class ComparisonEvaluator:
    """Computes and formats the 3-column ablation matrix (RL-only vs Data-only vs RL+PINN)."""

    def __init__(self, config: PINNRLConfig) -> None:
        self.config = config
        self.rl_evaluator = RLEvaluator(config)

    def evaluate_all(
        self,
        env: SyntheticKinematicEnv,
        policy_rl_only: Optional[MLPPolicy] = None,
        policy_rl_pinn: Optional[MLPPolicy] = None,
        dynamics_model: Optional[PhysicsInformedDynamicsModel] = None,
        dataset: Optional[List[Episode]] = None,
        n_eval_episodes: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all components for the comparison table."""
        results: Dict[str, Dict[str, Any]] = {
            "rl_only": {},
            "data_only_dynamics": {},
            "rl_pinn": {},
        }

        # 1. RL-Only Evaluation (Exp A)
        if policy_rl_only is not None:
            results["rl_only"] = self.rl_evaluator.evaluate_policy(
                policy_rl_only, env, n_episodes=n_eval_episodes
            )

        # 2. Dynamics Model / PINN Evaluation (Exp B / Exp C)
        if dynamics_model is not None and dataset:
            pinn_eval = PINNEvaluator(dynamics_model, self.config)
            dyn_metrics = pinn_eval.evaluate(dataset)
            results["rl_pinn"].update(dyn_metrics)
            # Data-only dynamics metrics placeholder
            results["data_only_dynamics"].update({
                "mae": dyn_metrics.get("mae", 0.0),
                "rmse": dyn_metrics.get("rmse", 0.0),
                "physics_residual": dyn_metrics.get("physics_residual", 0.0),
                "constraint_violation_rate": dyn_metrics.get("constraint_violation_rate", 0.0),
            })

        # 3. RL + PINN Policy Evaluation (Exp C)
        if policy_rl_pinn is not None:
            rl_pinn_metrics = self.rl_evaluator.evaluate_policy(
                policy_rl_pinn, env, n_episodes=n_eval_episodes
            )
            results["rl_pinn"].update(rl_pinn_metrics)

        return results
