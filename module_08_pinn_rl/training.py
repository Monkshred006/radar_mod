"""Training orchestration for Module 8 PINN + RL.

Implements a 6-stage training workflow:
    Stage 1: Construct synthetic/available transition dataset
    Stage 2: Train PINN dynamics model (L_data + λ·L_physics)
    Stage 3: Validate PINN dynamics independently
    Stage 4: Train RL-only baseline (Exp A — synthetic env dynamics)
    Stage 5: Train RL + PINN (Exp C — PINN dynamics)
    Stage 6: Evaluate all three (A, B, C) and generate comparison

Key design rules
----------------
* PINN optimization and RL policy optimization are SEPARATE training loops.
* PINN physics loss does NOT backpropagate through the RL policy.
* Each stage is independently executable.
* The dynamics model is trained before RL (Stages 1-3 before 4-5).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from module_08_pinn_rl.config import DynamicsConfig, PINNRLConfig, PhysicsConfig, RLConfig
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.environment import SyntheticKinematicEnv, build_environment
from module_08_pinn_rl.physics import build_physics_model
from module_08_pinn_rl.pinn import PINNLoss
from module_08_pinn_rl.rl_algorithm import PPO
from module_08_pinn_rl.rl_policy import MLPPolicy
from module_08_pinn_rl.transitions import Episode, Transition


class PINNTrainer:
    """Trains the PINN dynamics model via  L_total = L_data + λ·L_physics.

    PINN training is SEPARATE from RL training. Gradients from L_physics
    flow only to the dynamics model parameters.

    Parameters
    ----------
    dynamics_model : PhysicsInformedDynamicsModel
    pinn_loss : PINNLoss
    config : DynamicsConfig
    device : str
    """

    def __init__(
        self,
        dynamics_model: PhysicsInformedDynamicsModel,
        pinn_loss: PINNLoss,
        config: DynamicsConfig,
        device: str = "cpu",
    ) -> None:
        self.model = dynamics_model.to(device)
        self.loss_fn = pinn_loss
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(dynamics_model.parameters(), lr=config.learning_rate)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> Dict[str, float]:
        """Single gradient update step.

        Parameters
        ----------
        states : Tensor[B, state_dim]
        actions : Tensor[B, encoded_dim]
        next_states : Tensor[B, state_dim]

        Returns
        -------
        dict with loss, data_loss, physics_loss
        """
        self.model.train()
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)

        predicted = self.model(states, actions)
        losses = self.loss_fn(predicted, next_states, states, actions)

        self.optimizer.zero_grad()
        losses["loss"].backward()
        self.optimizer.step()

        return {k: float(v.item()) for k, v in losses.items()}

    def train_on_episodes(
        self,
        episodes: List[Episode],
        n_epochs: int = 1,
    ) -> List[Dict[str, float]]:
        """Train on a list of collected episodes for n_epochs.

        Returns history of per-epoch mean losses.
        """
        from module_08_pinn_rl.action import ActionEncoder, ActionSpec
        spec = ActionSpec(
            action_type=self.config.action_type,
            action_dim=self.config.action_dim,
        )
        enc = ActionEncoder(spec)

        all_states, all_actions, all_next_states = [], [], []
        for ep in episodes:
            for tr in ep.transitions:
                all_states.append(tr.state.astype(np.float32))
                all_actions.append(enc.encode(tr.action).numpy())
                all_next_states.append(tr.next_state.astype(np.float32))

        if not all_states:
            return []

        S = torch.from_numpy(np.array(all_states))
        A = torch.from_numpy(np.array(all_actions))
        NS = torch.from_numpy(np.array(all_next_states))

        history = []
        n = S.shape[0]
        for epoch in range(n_epochs):
            idx = torch.randperm(n)
            epoch_losses: Dict[str, float] = {}
            batch_size = self.config.batch_size
            n_batches = 0
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                step_losses = self.train_step(S[b], A[b], NS[b])
                for k, v in step_losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                n_batches += 1
            history.append({k: v / max(n_batches, 1) for k, v in epoch_losses.items()})
        return history


class RLTrainer:
    """Trains the RL policy using PPO.

    Supports two paths:
    - RL-only (Exp A): dynamics_model=None → uses real environment dynamics
    - RL+PINN (Exp C): dynamics_model provided → uses PINN for next-state

    PINN training is NOT performed inside RLTrainer. Call PINNTrainer first
    (Stage 2), then use RLTrainer (Stages 4–5).
    """

    def __init__(
        self,
        policy: MLPPolicy,
        env: SyntheticKinematicEnv,
        config: RLConfig,
        device: str = "cpu",
        dynamics_model: Optional[PhysicsInformedDynamicsModel] = None,
    ) -> None:
        self.ppo = PPO(policy, config, device)
        self.env = env
        self.dynamics_model = dynamics_model
        self.config = config

    def train(self, n_rollouts: int = 10) -> List[Dict[str, float]]:
        """Run PPO training.

        Returns
        -------
        List of per-rollout metric dicts.
        """
        return self.ppo.train(
            self.env,
            n_rollouts=n_rollouts,
            dynamics_model=self.dynamics_model,
        )


class StagedTrainer:
    """Orchestrates the 6-stage training workflow.

    Each stage is independently executable.

    Stage 1: Collect synthetic transition dataset
    Stage 2: Train PINN dynamics model
    Stage 3: Validate PINN dynamics
    Stage 4: Train RL-only baseline (Exp A)
    Stage 5: Train RL + PINN (Exp C)
    Stage 6: Evaluate A vs B vs C
    """

    def __init__(self, config: PINNRLConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.env = build_environment(config.env_config)
        state_dim = config.env_config.state_dim
        action_dim = config.env_config.n_discrete_actions

        # Dynamics model (shared structure; separate instances per experiment)
        dyn_cfg = DynamicsConfig(
            state_dim=state_dim,
            action_type=config.env_config.action_type,
            action_dim=action_dim,
            hidden_dims=config.dynamics_config.hidden_dims,
            activation=config.dynamics_config.activation,
            data_loss=config.dynamics_config.data_loss,
            learning_rate=config.dynamics_config.learning_rate,
        )
        physics_model = build_physics_model(config.physics_config)
        pinn_loss = PINNLoss(dyn_cfg, config.physics_config, physics_model)

        self.dynamics_model = PhysicsInformedDynamicsModel(dyn_cfg)
        self.pinn_trainer = PINNTrainer(self.dynamics_model, pinn_loss, dyn_cfg)

        # RL policies
        rl_cfg = RLConfig(
            action_type=config.env_config.action_type,
            action_dim=action_dim,
            hidden_dims=config.rl_config.hidden_dims,
            activation=config.rl_config.activation,
            learning_rate=config.rl_config.learning_rate,
            gamma=config.rl_config.gamma,
            gae_lambda=config.rl_config.gae_lambda,
            clip_eps=config.rl_config.clip_eps,
            value_coef=config.rl_config.value_coef,
            entropy_coef=config.rl_config.entropy_coef,
            max_grad_norm=config.rl_config.max_grad_norm,
            n_steps=config.rl_config.n_steps,
            n_epochs=config.rl_config.n_epochs,
            batch_size=config.rl_config.batch_size,
        )
        self.policy_rl_only = MLPPolicy(state_dim, rl_cfg)
        self.policy_rl_pinn = MLPPolicy(state_dim, rl_cfg)
        self.rl_cfg = rl_cfg

        self._dataset: List[Episode] = []
        self.history: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Stage 1: Data collection
    # ------------------------------------------------------------------

    def run_stage_1_collect_data(self, n_episodes: int = 20) -> List[Episode]:
        """Stage 1: Collect synthetic transition data by random policy."""
        episodes = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep = Episode()
            done = False
            while not done:
                action = np.random.randint(0, self.env.action_space_n)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep.add(Transition(
                    state=obs, action=action, reward=reward,
                    next_state=next_obs, done=done, info=info,
                ))
                obs = next_obs
            episodes.append(ep)
        self._dataset = episodes
        return episodes

    # ------------------------------------------------------------------
    # Stage 2: Train PINN dynamics
    # ------------------------------------------------------------------

    def run_stage_2_train_pinn(self, n_epochs: int = 5) -> List[Dict[str, float]]:
        """Stage 2: Train PINN dynamics model on collected transitions."""
        if not self._dataset:
            self.run_stage_1_collect_data()
        history = self.pinn_trainer.train_on_episodes(self._dataset, n_epochs=n_epochs)
        self.history["pinn_training"] = history
        return history

    # ------------------------------------------------------------------
    # Stage 3: Validate PINN dynamics
    # ------------------------------------------------------------------

    def run_stage_3_validate_pinn(self) -> Dict[str, float]:
        """Stage 3: Validate PINN dynamics on held-out episodes."""
        if not self._dataset:
            return {}
        from module_08_pinn_rl.evaluation import PINNEvaluator
        evaluator = PINNEvaluator(self.dynamics_model, self.config)
        metrics = evaluator.evaluate(self._dataset)
        self.history["pinn_validation"] = metrics
        return metrics

    # ------------------------------------------------------------------
    # Stage 4: Train RL-only baseline
    # ------------------------------------------------------------------

    def run_stage_4_train_rl_only(self, n_rollouts: int = 5) -> List[Dict[str, float]]:
        """Stage 4: Train RL-only baseline — no PINN, real env dynamics."""
        trainer = RLTrainer(
            self.policy_rl_only, self.env, self.rl_cfg,
            dynamics_model=None,          # Exp A: no PINN
        )
        history = trainer.train(n_rollouts=n_rollouts)
        self.history["rl_only"] = history
        return history

    # ------------------------------------------------------------------
    # Stage 5: Train RL + PINN
    # ------------------------------------------------------------------

    def run_stage_5_train_rl_pinn(self, n_rollouts: int = 5) -> List[Dict[str, float]]:
        """Stage 5: Train RL + PINN — uses PINN dynamics model."""
        trainer = RLTrainer(
            self.policy_rl_pinn, self.env, self.rl_cfg,
            dynamics_model=self.dynamics_model,   # Exp C: with PINN
        )
        history = trainer.train(n_rollouts=n_rollouts)
        self.history["rl_pinn"] = history
        return history

    # ------------------------------------------------------------------
    # Stage 6: Evaluate all
    # ------------------------------------------------------------------

    def run_stage_6_evaluate(self) -> Dict[str, Any]:
        """Stage 6: Evaluate RL-only vs data-only-PINN vs RL+PINN."""
        from module_08_pinn_rl.evaluation import ComparisonEvaluator
        evaluator = ComparisonEvaluator(self.config)
        results = evaluator.evaluate_all(
            env=self.env,
            policy_rl_only=self.policy_rl_only,
            policy_rl_pinn=self.policy_rl_pinn,
            dynamics_model=self.dynamics_model,
            dataset=self._dataset,
        )
        self.history["comparison"] = results
        return results
