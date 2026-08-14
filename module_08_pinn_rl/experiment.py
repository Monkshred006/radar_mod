"""Experiment runner and comparison matrix generator for Module 8 PINN + RL.

Executes:
- Experiment A: RL-only (no PINN, ordinary/synthetic env dynamics)
- Experiment B: Data-only dynamics (PINN trained with lambda_phys = 0)
- Experiment C: RL + PINN (PINN trained with lambda_phys > 0)

Formats results into a verified comparison table. All results from synthetic
environments are explicitly labeled as SYNTHETIC VERIFICATION.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from module_08_pinn_rl.baselines import DataOnlyDynamics, RLOnlyBaseline, RLPINNSystem
from module_08_pinn_rl.config import PINNRLConfig
from module_08_pinn_rl.evaluation import ComparisonEvaluator, PINNEvaluator, RLEvaluator
from module_08_pinn_rl.transitions import Episode, Transition


def collect_synthetic_dataset(env, n_episodes: int = 20, seed: int = 42) -> List[Episode]:
    """Collect transition dataset using random actions."""
    episodes = []
    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        ep = Episode()
        done = False
        while not done:
            action = np.random.randint(0, env.action_space_n)
            next_obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep.add(Transition(
                state=obs, action=action, reward=r,
                next_state=next_obs, done=done, info=info,
            ))
            obs = next_obs
        episodes.append(ep)
    return episodes


class ExperimentRunner:
    """Orchestrates Experiment A (RL-only), B (Data-only), and C (RL+PINN)."""

    def __init__(self, config: Optional[PINNRLConfig] = None) -> None:
        self.config = config or PINNRLConfig()

    def run_all_experiments(
        self,
        n_dataset_episodes: int = 20,
        n_pinn_epochs: int = 5,
        n_rl_rollouts: int = 5,
        n_eval_episodes: int = 10,
    ) -> Dict[str, Any]:
        """Execute full experimental matrix."""
        # 1. Instantiate systems
        sys_a = RLOnlyBaseline(self.config)
        sys_b = DataOnlyDynamics(self.config)
        sys_c = RLPINNSystem(self.config)

        # 2. Collect dataset for dynamics training
        dataset = collect_synthetic_dataset(
            sys_a.env, n_episodes=n_dataset_episodes, seed=self.config.seed
        )

        # 3. Train models
        # Exp A: RL-only
        _ = sys_a.train(n_rollouts=n_rl_rollouts)

        # Exp B: Data-only dynamics (lambda_phys = 0)
        _ = sys_b.train(dataset, n_epochs=n_pinn_epochs)

        # Exp C: RL + PINN
        _ = sys_c.train_pinn(dataset, n_epochs=n_pinn_epochs)
        _ = sys_c.train_rl(n_rollouts=n_rl_rollouts)

        # 4. Evaluate each experiment
        rl_eval = RLEvaluator(self.config)
        eval_a = rl_eval.evaluate_policy(sys_a.policy, sys_a.env, n_episodes=n_eval_episodes)

        pinn_eval_b = PINNEvaluator(sys_b.dynamics_model, self.config)
        eval_b = pinn_eval_b.evaluate(dataset)

        pinn_eval_c = PINNEvaluator(sys_c.dynamics_model, self.config)
        eval_c_dyn = pinn_eval_c.evaluate(dataset)
        eval_c_rl = rl_eval.evaluate_policy(sys_c.policy, sys_c.env, n_episodes=n_eval_episodes)

        results = {
            "exp_a_rl_only": eval_a,
            "exp_b_data_only": eval_b,
            "exp_c_rl_pinn": {**eval_c_dyn, **eval_c_rl},
        }

        return results

    def generate_comparison_matrix(self, results: Dict[str, Any]) -> str:
        """Format 3-column markdown table comparing Exp A, Exp B, and Exp C."""
        res_a = results.get("exp_a_rl_only", {})
        res_b = results.get("exp_b_data_only", {})
        res_c = results.get("exp_c_rl_pinn", {})

        def fmt(val: Any, unit: str = "", default: str = "N/A") -> str:
            if val is None or val == "N/A":
                return default
            if isinstance(val, (float, np.floating)):
                return f"{val:.4f}{unit}"
            if isinstance(val, (int, np.integer)):
                return f"{val}{unit}"
            return str(val)

        lines = [
            "# Module 8 Ablation Comparison Matrix (SYNTHETIC VERIFICATION)",
            "",
            "> **Disclaimer**: The results below are generated using `SyntheticKinematicEnv` and "
            "`KinematicPhysicsModel` for software verification only. They do not represent real "
            "PhotonShield hardware performance.",
            "",
            "| Metric | RL-only (Exp A) | Data-only Dynamics (Exp B) | RL + PINN (Exp C) |",
            "|---|---|---|---|",
            f"| **Episode Reward (Mean)** | {fmt(res_a.get('mean_reward'))} | N/A | {fmt(res_c.get('mean_reward'))} |",
            f"| **Success Rate** | {fmt(res_a.get('success_rate'))} | N/A | {fmt(res_c.get('success_rate'))} |",
            f"| **State Prediction Error (MAE)** | N/A | {fmt(res_b.get('mae'))} | {fmt(res_c.get('mae'))} |",
            f"| **State Prediction Error (RMSE)** | N/A | {fmt(res_b.get('rmse'))} | {fmt(res_c.get('rmse'))} |",
            f"| **Physics Residual** | N/A | {fmt(res_b.get('physics_residual'))} | {fmt(res_c.get('physics_residual'))} |",
            f"| **Constraint Violation Rate** | N/A | {fmt(res_b.get('constraint_violation_rate'))} | {fmt(res_c.get('constraint_violation_rate'))} |",
            f"| **Episode Length (Mean)** | {fmt(res_a.get('mean_length'))} | N/A | {fmt(res_c.get('mean_length'))} |",
            f"| **Action Cost (Mean)** | {fmt(res_a.get('mean_action_cost'))} | N/A | {fmt(res_c.get('mean_action_cost'))} |",
            f"| **Inference Latency (ms)** | {fmt(res_a.get('inference_latency_ms'), ' ms')} | N/A | {fmt(res_c.get('inference_latency_ms'), ' ms')} |",
            "",
        ]
        return "\n".join(lines)
