"""Tests for Module 8 RL State construction and dimension derivation."""

import pytest
import torch

from module_07_decision.outputs import PhotonShieldDecisionOutput
from module_08_pinn_rl.config import RLStateConfig
from module_08_pinn_rl.state import RLState, RLStateBuilder


class TestRLState:
    def test_state_dim_derived_default(self):
        cfg = RLStateConfig()  # 128 + 1 + 1 + 3 = 133
        assert cfg.state_dim == 133

    def test_state_dim_derived_custom_dmodel(self):
        cfg64 = RLStateConfig(mamba_latent_dim=64)
        assert cfg64.state_dim == 64 + 1 + 1 + 3  # 69

        cfg256 = RLStateConfig(mamba_latent_dim=256)
        assert cfg256.state_dim == 256 + 1 + 1 + 3  # 261

    def test_state_dim_with_physical_state(self):
        cfg = RLStateConfig(
            mamba_latent_dim=128,
            use_raw_physical_state=True,
            physical_state_dim=6,
        )
        assert cfg.state_dim == 128 + 1 + 1 + 3 + 6  # 139

    def test_build_state_from_tensors(self):
        cfg = RLStateConfig(mamba_latent_dim=128)
        builder = RLStateBuilder(cfg)

        z = torch.randn(128)
        state = builder.build(
            pooled_output=z,
            target_probability=0.85,
            anomaly_probability=0.12,
            environment_output=[25.0, 60.0, 1013.0],
        )

        assert state.vector.shape == (133,)
        assert state.components["mamba_latent"].shape == (128,)
        assert state.components["target_probability"].item() == pytest.approx(0.85)
        assert state.components["anomaly_probability"].item() == pytest.approx(0.12)
        assert state.components["environment_output"].shape == (3,)

    def test_build_from_decision_output(self):
        cfg = RLStateConfig(mamba_latent_dim=128)
        builder = RLStateBuilder(cfg)

        z = torch.randn(128)
        dec_out = PhotonShieldDecisionOutput(
            target_probability=0.92,
            anomaly_probability=0.03,
            environmental_assessment=[22.5, 55.0, 1012.0],
            target_detected=True,
            anomaly_detected=False,
        )

        state = builder.build_from_decision_output(
            pooled_output=z,
            decision_output=dec_out,
        )

        assert state.vector.shape == (133,)
        assert state.vector[128].item() == pytest.approx(0.92)  # target prob
        assert state.vector[129].item() == pytest.approx(0.03)  # anomaly prob

    def test_immutability_and_cloning(self):
        cfg = RLStateConfig(mamba_latent_dim=64)
        builder = RLStateBuilder(cfg)

        z1 = torch.ones(64)
        s1 = builder.build(z1, 0.5, 0.5, [1.0, 2.0, 3.0])

        z2 = torch.zeros(64)
        s2 = builder.build(z2, 0.0, 0.0, [0.0, 0.0, 0.0])

        # Modifying z1 tensor should not alter s1 vector
        z1.zero_()
        assert s1.vector[0].item() == 1.0
        assert s2.vector[0].item() == 0.0
