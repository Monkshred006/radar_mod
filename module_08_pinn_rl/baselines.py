"""Baseline implementations for Module 8 ablation comparison.

Provides three distinct system configurations:

1. RLOnlyBaseline (Experiment A):
   - No PINN dynamics model.
   - RL policy interacts directly with the ordinary / synthetic environment dynamics.
   - No physics-informed loss.

2. DataOnlyDynamics (Experiment B):
   - PINN / dynamics model trained using data loss ONLY (lambda_phys = 0).
   - Physics residual is disabled.
   - This is NOT the RL-only baseline.

3. RLPINNSystem (Experiment C):
   - PINN-based dynamics model trained with:
     L_total = L_data + lambda_phys * L_physics (lambda_phys > 0).
   - RL policy trained and/or evaluated using the PINN dynamics model.

All three configurations share:
- Same initial conditions
- Same action space
- Same task
- Same reward definition
- Same evaluation protocol
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from module_08_pinn_rl.config import DynamicsConfig, PINNRLConfig, PhysicsConfig, RLConfig
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.environment import SyntheticKinematicEnv, build_environment
from module_08_pinn_rl.physics import NoPhysicsModel, build_physics_model
from module_08_pinn_rl.pinn import PINNLoss
from module_08_pinn_rl.rl_algorithm import PPO
from module_08_pinn_rl.rl_policy import MLPPolicy
from module_08_pinn_rl.training import PINNTrainer, RLTrainer
from module_08_pinn_rl.transitions import Episode


class RLOnlyBaseline:
    """Experiment A: Pure RL baseline with NO PINN dynamics model."""

    def __init__(self, config: PINNRLConfig) -> None:
        self.config = config
        self.env = build_environment(config.env_config)
        state_dim = config.env_config.state_dim
        action_dim = config.env_config.n_discrete_actions

        rl_cfg = RLConfig(
            action_type=config.env_config.action_type,
            action_dim=action_dim,
            hidden_dims=config.rl_config.hidden_dims,
            activation=config.rl_config.activation,
            learning_rate=config.rl_config.learning_rate,
            n_steps=config.rl_config.n_steps,
            n_epochs=config.rl_config.n_epochs,
            batch_size=config.rl_config.batch_size,
        )
        self.policy = MLPPolicy(state_dim, rl_cfg)
        self.trainer = RLTrainer(self.policy, self.env, rl_cfg, dynamics_model=None)

    def train(self, n_rollouts: int = 10) -> List[Dict[str, float]]:
        return self.trainer.train(n_rollouts=n_rollouts)


class DataOnlyDynamics:
    """Experiment B: Dynamics model trained with DATA LOSS ONLY (lambda_phys = 0)."""

    def __init__(self, config: PINNRLConfig) -> None:
        self.config = config
        state_dim = config.env_config.state_dim
        action_dim = config.env_config.n_discrete_actions

        dyn_cfg = DynamicsConfig(
            state_dim=state_dim,
            action_type=config.env_config.action_type,
            action_dim=action_dim,
            hidden_dims=config.dynamics_config.hidden_dims,
            activation=config.dynamics_config.activation,
            data_loss=config.dynamics_config.data_loss,
            learning_rate=config.dynamics_config.learning_rate,
        )
        # Data-only uses NoPhysicsModel or lambda_physics = 0.0
        no_phys_cfg = PhysicsConfig(physics_model="none", lambda_physics=0.0)
        pinn_loss = PINNLoss(dyn_cfg, no_phys_cfg, NoPhysicsModel())

        self.dynamics_model = PhysicsInformedDynamicsModel(dyn_cfg)
        self.trainer = PINNTrainer(self.dynamics_model, pinn_loss, dyn_cfg)

    def train(self, episodes: List[Episode], n_epochs: int = 5) -> List[Dict[str, float]]:
        return self.trainer.train_on_episodes(episodes, n_epochs=n_epochs)


class RLPINNSystem:
    """Experiment C: RL + Physics-Informed Neural Network (lambda_phys > 0)."""

    def __init__(self, config: PINNRLConfig) -> None:
        self.config = config
        self.env = build_environment(config.env_config)
        state_dim = config.env_config.state_dim
        action_dim = config.env_config.n_discrete_actions

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

        rl_cfg = RLConfig(
            action_type=config.env_config.action_type,
            action_dim=action_dim,
            hidden_dims=config.rl_config.hidden_dims,
            activation=config.rl_config.activation,
            learning_rate=config.rl_config.learning_rate,
            n_steps=config.rl_config.n_steps,
            n_epochs=config.rl_config.n_epochs,
            batch_size=config.rl_config.batch_size,
        )
        self.policy = MLPPolicy(state_dim, rl_cfg)
        self.rl_trainer = RLTrainer(
            self.policy, self.env, rl_cfg, dynamics_model=self.dynamics_model
        )

    def train_pinn(self, episodes: List[Episode], n_epochs: int = 5) -> List[Dict[str, float]]:
        return self.pinn_trainer.train_on_episodes(episodes, n_epochs=n_epochs)

    def train_rl(self, n_rollouts: int = 10) -> List[Dict[str, float]]:
        return self.rl_trainer.train(n_rollouts=n_rollouts)
