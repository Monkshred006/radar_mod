"""End-to-End integration tests for Module 8 PINN + RL."""

import pytest
import torch

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_07_decision.config import DecisionConfig, DecisionModelConfig
from module_07_decision.inference import PhotonShieldDecisionPipeline
from module_07_decision.multitask import PhotonShieldMultiTask
from module_08_pinn_rl.config import (
    DynamicsConfig,
    PINNRLConfig,
    PhysicsConfig,
    RLConfig,
    RLStateConfig,
)
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.pinn import PINNLoss
from module_08_pinn_rl.rl_policy import MLPPolicy
from module_08_pinn_rl.state import RLStateBuilder
from module_08_pinn_rl.training import PINNTrainer
from module_08_pinn_rl.transitions import Episode, Transition


class TestEndToEndModule08:
    def test_full_pipeline_module4_to_module7_to_module8(self):
        """Verify: Module 4 Engine -> Module 7 Decision Pipeline -> RLStateBuilder -> PINN -> Policy."""
        # 1. Module 4
        mamba_cfg = MambaHybridConfig(
            d_model=128,
            num_layers=1,
            num_attention_heads=2,
            head_config=TaskHeadConfig(head_type="classification"),
        )
        engine = PhotonMambaHybrid(mamba_cfg)
        engine.eval()

        # Synthetic Module 3 output
        m3_out = {
            "tokens": torch.randn(2, 4, 5, 48),
            "token_mask": torch.ones(2, 4, 5, 48, dtype=torch.bool),
            "features": torch.randn(2, 4, 101),
            "timestamps": torch.arange(4).float().unsqueeze(0).expand(2, -1),
        }

        with torch.no_grad():
            eng_out = engine(m3_out)
        pooled = eng_out["pooled_output"]  # [2, 128]

        # 2. Module 7
        dec_m_cfg = DecisionModelConfig(d_model=128)
        multi_task = PhotonShieldMultiTask(dec_m_cfg)
        dec_cfg = DecisionConfig()
        pipeline = PhotonShieldDecisionPipeline(multi_task, dec_cfg, engine=engine)

        dec_outputs = pipeline.predict_pooled(pooled)
        assert len(dec_outputs) == 2
        assert dec_outputs[0].pooled_output is not None

        # 3. Module 8 RLStateBuilder
        rl_state_cfg = RLStateConfig(mamba_latent_dim=128, environment_dim=3)
        state_builder = RLStateBuilder(rl_state_cfg)
        assert state_builder.state_dim == 133

        rl_state_0 = state_builder.build_from_decision_output(
            pooled_output=dec_outputs[0].pooled_output,
            decision_output=dec_outputs[0],
        )
        assert rl_state_0.vector.shape == (133,)

        # 4. Module 8 PINN Dynamics Model
        dyn_cfg = DynamicsConfig(state_dim=133, action_dim=4)
        pinn_dyn = PhysicsInformedDynamicsModel(dyn_cfg)

        action_enc = torch.zeros(1, 4)
        action_enc[0, 1] = 1.0  # Action 1

        with torch.no_grad():
            next_state_pred = pinn_dyn(rl_state_0.vector.unsqueeze(0), action_enc)
        assert next_state_pred.shape == (1, 133)

        # 5. Module 8 RL Policy
        rl_cfg = RLConfig(action_dim=4)
        policy = MLPPolicy(state_dim=133, config=rl_cfg)

        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(rl_state_0.vector)
        assert action.item() in [0, 1, 2, 3]

    def test_pinn_residual_decreases_with_training_on_synthetic_data(self):
        """Verify that PINN training reduces physics residual and data loss."""
        torch.manual_seed(42)

        dyn_cfg = DynamicsConfig(
            state_dim=2, action_dim=3, hidden_dims=[32, 32], learning_rate=5e-3
        )
        phys_cfg = PhysicsConfig(physics_model="kinematic", lambda_physics=0.5, dt=0.1)
        model = PhysicsInformedDynamicsModel(dyn_cfg)
        loss_fn = PINNLoss(dyn_cfg, phys_cfg)
        trainer = PINNTrainer(model, loss_fn, dyn_cfg)

        # Synthetic exact kinematic data
        # x_next = x + v * 0.1
        # v_next = v + a_force * 0.1
        force_map = {0: -1.0, 1: 0.0, 2: 1.0}

        episodes = []
        for _ in range(5):
            ep = Episode()
            x, v = 0.0, 0.5
            for _ in range(20):
                a_idx = int(torch.randint(0, 3, (1,)).item())
                f = force_map[a_idx]
                x_n = x + v * 0.1
                v_n = v + f * 0.1
                ep.add(Transition(
                    state=torch.tensor([x, v]).numpy(),
                    action=a_idx,
                    reward=0.0,
                    next_state=torch.tensor([x_n, v_n]).numpy(),
                    done=False,
                ))
                x, v = x_n, v_n
            episodes.append(ep)

        history = trainer.train_on_episodes(episodes, n_epochs=15)

        initial_loss = history[0]["loss"]
        final_loss = history[-1]["loss"]

        assert final_loss < initial_loss
        assert history[-1]["physics_loss"] <= history[0]["physics_loss"]
