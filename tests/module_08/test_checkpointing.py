"""Tests for PINN and RL checkpoint save/load roundtrip."""

import tempfile
from pathlib import Path

import pytest
import torch

from module_08_pinn_rl.checkpointing import (
    load_pinn_checkpoint,
    load_rl_checkpoint,
    save_pinn_checkpoint,
    save_rl_checkpoint,
)
from module_08_pinn_rl.config import DynamicsConfig, PINNRLConfig, RLConfig
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.rl_policy import MLPPolicy


class TestCheckpointing:
    def test_pinn_save_load_roundtrip(self):
        dyn_cfg = DynamicsConfig(state_dim=133, action_dim=4)
        model = PhysicsInformedDynamicsModel(dyn_cfg)
        config = PINNRLConfig(dynamics_config=dyn_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "pinn_ckpt.pt"
            save_pinn_checkpoint(model, ckpt_path, config=config, training_history=[{"loss": 0.1}])

            assert ckpt_path.exists()

            loaded_model, ckpt_dict = load_pinn_checkpoint(ckpt_path)
            assert ckpt_dict["training_history"] == [{"loss": 0.1}]

            dummy_s = torch.randn(2, 133)
            dummy_a = torch.zeros(2, 4)
            dummy_a[:, 0] = 1.0

            out_orig = model(dummy_s, dummy_a)
            out_loaded = loaded_model(dummy_s, dummy_a)

            assert torch.allclose(out_orig, out_loaded)

    def test_rl_save_load_roundtrip(self):
        rl_cfg = RLConfig(action_dim=3)
        policy = MLPPolicy(state_dim=10, config=rl_cfg)
        config = PINNRLConfig(rl_config=rl_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "rl_ckpt.pt"
            save_rl_checkpoint(policy, ckpt_path, state_dim=10, config=config)

            assert ckpt_path.exists()

            loaded_policy, _ = load_rl_checkpoint(ckpt_path)

            dummy_s = torch.randn(2, 10)
            logits_orig, val_orig = policy(dummy_s)
            logits_loaded, val_loaded = loaded_policy(dummy_s)

            assert torch.allclose(logits_orig, logits_loaded)
            assert torch.allclose(val_orig, val_loaded)
            assert policy.act(dummy_s[0]) == loaded_policy.act(dummy_s[0])
