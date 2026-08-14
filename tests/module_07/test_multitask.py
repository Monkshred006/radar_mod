"""Tests for PhotonShieldMultiTask container and MultiTaskDecisionLoss."""

import pytest
import torch
from module_07_decision.config import DecisionModelConfig
from module_07_decision.multitask import PhotonShieldMultiTask, MultiTaskDecisionLoss


class TestMultiTask:
    def test_multitask_outputs_keys(self):
        cfg = DecisionModelConfig(d_model=32, enable_target=True, enable_anomaly=True, enable_environment=True)
        model = PhotonShieldMultiTask(cfg)

        x = torch.randn(4, 32)
        outs = model(x)
        assert "target_logits" in outs
        assert "anomaly_logits" in outs
        assert "environment_output" in outs

        assert outs["target_logits"].shape == (4, 2)
        assert outs["anomaly_logits"].shape == (4, 1)
        assert outs["environment_output"].shape == (4, 3)

    def test_disabled_heads(self):
        cfg = DecisionModelConfig(d_model=32, enable_target=True, enable_anomaly=False, enable_environment=False)
        model = PhotonShieldMultiTask(cfg)
        outs = model(torch.randn(2, 32))
        assert "target_logits" in outs
        assert "anomaly_logits" not in outs
        assert "environment_output" not in outs

    def test_multitask_loss_calculation(self):
        cfg = DecisionModelConfig(d_model=32)
        model = PhotonShieldMultiTask(cfg)
        loss_fn = MultiTaskDecisionLoss(cfg)

        outs = model(torch.randn(4, 32))
        targets = {
            "target_labels": torch.tensor([0, 1, 0, 1]),
            "anomaly_labels": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "environment_labels": torch.randn(4, 3),
        }

        losses = loss_fn(outs, targets)
        assert "loss" in losses
        assert torch.isfinite(losses["loss"])

    def test_missing_label_masking(self):
        """Missing labels (-100 or -1.0) must be masked out and not cause crash or NaN loss."""
        cfg = DecisionModelConfig(d_model=32)
        model = PhotonShieldMultiTask(cfg)
        loss_fn = MultiTaskDecisionLoss(cfg)

        outs = model(torch.randn(4, 32))
        targets = {
            "target_labels": torch.tensor([-100, 1, -100, 0]),  # 2 missing labels
            "anomaly_labels": torch.tensor([-1.0, 1.0, -1.0, 0.0]),  # 2 missing labels
            "environment_labels": torch.tensor([[1.0, 2.0, 3.0], [float("nan"), float("nan"), float("nan")], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        }

        losses = loss_fn(outs, targets)
        assert torch.isfinite(losses["loss"])
