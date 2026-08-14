"""Tests for baseline experiments A, B, and C."""

import pytest

from module_08_pinn_rl.baselines import DataOnlyDynamics, RLOnlyBaseline, RLPINNSystem
from module_08_pinn_rl.config import PINNRLConfig
from module_08_pinn_rl.experiment import ExperimentRunner, collect_synthetic_dataset


class TestBaselines:
    def test_rl_only_baseline(self):
        config = PINNRLConfig()
        config.rl_config.n_steps = 16
        config.rl_config.batch_size = 8

        baseline = RLOnlyBaseline(config)
        history = baseline.train(n_rollouts=2)

        assert len(history) == 2
        assert "mean_reward" in history[0]

    def test_data_only_dynamics(self):
        config = PINNRLConfig()
        baseline = DataOnlyDynamics(config)

        dataset = collect_synthetic_dataset(
            baseline.trainer.model.predict,  # dummy env placeholder
            n_episodes=0,
        )
        # Directly test with synthetic transitions
        from module_08_pinn_rl.environment import build_environment
        env = build_environment(config.env_config)
        dataset = collect_synthetic_dataset(env, n_episodes=2)

        hist = baseline.train(dataset, n_epochs=1)
        assert len(hist) >= 1
        assert "loss" in hist[0]
        # For data-only, loss equals data_loss
        assert hist[0]["loss"] == pytest.approx(hist[0]["data_loss"], rel=1e-5)

    def test_rl_pinn_system(self):
        config = PINNRLConfig()
        config.rl_config.n_steps = 16
        config.rl_config.batch_size = 8

        system = RLPINNSystem(config)
        dataset = collect_synthetic_dataset(system.env, n_episodes=2)

        pinn_hist = system.train_pinn(dataset, n_epochs=1)
        assert len(pinn_hist) >= 1

        rl_hist = system.train_rl(n_rollouts=1)
        assert len(rl_hist) == 1

    def test_experiment_runner_comparison_matrix(self):
        config = PINNRLConfig()
        config.rl_config.n_steps = 16
        config.rl_config.batch_size = 8

        runner = ExperimentRunner(config)
        results = runner.run_all_experiments(
            n_dataset_episodes=2,
            n_pinn_epochs=1,
            n_rl_rollouts=1,
            n_eval_episodes=2,
        )

        assert "exp_a_rl_only" in results
        assert "exp_b_data_only" in results
        assert "exp_c_rl_pinn" in results

        table_md = runner.generate_comparison_matrix(results)
        assert "RL-only (Exp A)" in table_md
        assert "Data-only Dynamics (Exp B)" in table_md
        assert "RL + PINN (Exp C)" in table_md
        assert "SYNTHETIC VERIFICATION" in table_md
