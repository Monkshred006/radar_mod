"""Tests for training workflows and StagedTrainer."""

import pytest
import torch

from module_08_pinn_rl.config import DynamicsConfig, PINNRLConfig, PhysicsConfig, RLConfig
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.environment import SyntheticKinematicEnv
from module_08_pinn_rl.pinn import PINNLoss
from module_08_pinn_rl.rl_policy import MLPPolicy
from module_08_pinn_rl.training import PINNTrainer, RLTrainer, StagedTrainer
from module_08_pinn_rl.transitions import Episode, Transition


class TestTraining:
    def test_pinn_trainer_step_updates_weights(self):
        dyn_cfg = DynamicsConfig(state_dim=2, action_dim=3, learning_rate=1e-2)
        phys_cfg = PhysicsConfig(physics_model="kinematic", lambda_physics=0.1)
        model = PhysicsInformedDynamicsModel(dyn_cfg)
        loss_fn = PINNLoss(dyn_cfg, phys_cfg)
        trainer = PINNTrainer(model, loss_fn, dyn_cfg)

        init_param = next(model.parameters()).clone()

        s = torch.randn(16, 2)
        a = torch.zeros(16, 3)
        a[:, 0] = 1.0
        ns = torch.randn(16, 2)

        res = trainer.train_step(s, a, ns)
        assert "loss" in res

        updated_param = next(model.parameters())
        assert not torch.equal(init_param, updated_param)

    def test_rl_trainer_runs_and_returns_metrics(self):
        rl_cfg = RLConfig(action_dim=3, n_steps=16, n_epochs=2, batch_size=8)
        policy = MLPPolicy(state_dim=2, config=rl_cfg)
        config = PINNRLConfig(rl_config=rl_cfg)
        env = SyntheticKinematicEnv(config.env_config)

        trainer = RLTrainer(policy, env, rl_cfg)
        history = trainer.train(n_rollouts=2)

        assert len(history) == 2
        assert "mean_reward" in history[0]

    def test_staged_trainer_all_stages(self):
        config = PINNRLConfig()
        config.rl_config.n_steps = 16
        config.rl_config.batch_size = 8
        trainer = StagedTrainer(config)

        # Stage 1
        dataset = trainer.run_stage_1_collect_data(n_episodes=2)
        assert len(dataset) == 2

        # Stage 2
        pinn_hist = trainer.run_stage_2_train_pinn(n_epochs=1)
        assert len(pinn_hist) >= 1

        # Stage 3
        pinn_val = trainer.run_stage_3_validate_pinn()
        assert "mae" in pinn_val

        # Stage 4
        rl_only_hist = trainer.run_stage_4_train_rl_only(n_rollouts=1)
        assert len(rl_only_hist) == 1

        # Stage 5
        rl_pinn_hist = trainer.run_stage_5_train_rl_pinn(n_rollouts=1)
        assert len(rl_pinn_hist) == 1

        # Stage 6
        comparison = trainer.run_stage_6_evaluate()
        assert "rl_only" in comparison
        assert "rl_pinn" in comparison
