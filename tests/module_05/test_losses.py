"""Tests for Module 5 loss functions."""

import pytest
import torch

from module_05_training.losses import (
    get_loss_fn,
    SafeLoss,
    TrainingNaNError,
    WeightedMultiTaskLoss,
)
from module_05_training.config import TrainingConfig


def _cfg(loss_name: str, target_type="regression") -> TrainingConfig:
    return TrainingConfig(loss_name=loss_name, target_type=target_type)


class TestGetLossFn:
    def test_mse(self):
        fn = get_loss_fn(_cfg("mse"))
        pred = torch.tensor([[1.0, 2.0]])
        tgt = torch.tensor([[1.0, 2.0]])
        loss = fn(pred, tgt)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_l1(self):
        fn = get_loss_fn(_cfg("l1"))
        pred = torch.tensor([[2.0]])
        tgt = torch.tensor([[0.0]])
        assert fn(pred, tgt).item() == pytest.approx(2.0, abs=1e-5)

    def test_smooth_l1(self):
        fn = get_loss_fn(_cfg("smooth_l1"))
        pred = torch.tensor([[1.0]])
        tgt = torch.tensor([[0.0]])
        loss = fn(pred, tgt)
        assert torch.isfinite(loss)

    def test_cross_entropy(self):
        fn = get_loss_fn(_cfg("cross_entropy", "classification"))
        logits = torch.tensor([[1.0, 0.0]])
        labels = torch.tensor([0])
        loss = fn(logits, labels)
        assert torch.isfinite(loss)

    def test_bce_with_logits(self):
        fn = get_loss_fn(_cfg("bce_with_logits"))
        logits = torch.tensor([[0.5]])
        targets = torch.tensor([[1.0]])
        loss = fn(logits, targets)
        assert torch.isfinite(loss)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss_fn(_cfg("quantum_entropy"))


class TestNaNDetection:
    def test_nan_loss_raises(self):
        fn = get_loss_fn(_cfg("mse"))
        pred = torch.tensor([[float("nan")]])
        tgt = torch.tensor([[1.0]])
        with pytest.raises(TrainingNaNError):
            fn(pred, tgt)

    def test_inf_loss_raises(self):
        fn = get_loss_fn(_cfg("mse"))
        pred = torch.tensor([[float("inf")]])
        tgt = torch.tensor([[0.0]])
        with pytest.raises(TrainingNaNError):
            fn(pred, tgt)

    def test_finite_loss_ok(self):
        fn = get_loss_fn(_cfg("mse"))
        pred = torch.tensor([[1.5]])
        tgt = torch.tensor([[1.0]])
        loss = fn(pred, tgt)
        assert torch.isfinite(loss)


class TestWeightedMultiTaskLoss:
    def test_multitask_sum(self):
        import torch.nn as nn
        task_losses = {"cls": nn.CrossEntropyLoss(), "reg": nn.MSELoss()}
        weights = {"cls": 1.0, "reg": 0.5}
        mtl = WeightedMultiTaskLoss(task_losses, weights)

        preds = {
            "cls": torch.tensor([[1.0, 0.0]]),
            "reg": torch.tensor([[1.0]]),
        }
        targets = {
            "cls": torch.tensor([0]),
            "reg": torch.tensor([[1.0]]),
        }
        total, per_task = mtl(preds, targets)
        assert torch.isfinite(total)
        assert "cls" in per_task and "reg" in per_task

    def test_multitask_missing_task_skipped(self):
        import torch.nn as nn
        task_losses = {"cls": nn.CrossEntropyLoss(), "reg": nn.MSELoss()}
        weights = {"cls": 1.0, "reg": 1.0}
        mtl = WeightedMultiTaskLoss(task_losses, weights)

        preds = {"cls": torch.tensor([[1.0, 0.0]])}
        targets = {"cls": torch.tensor([0])}
        total, per_task = mtl(preds, targets)
        assert torch.isfinite(total)
        assert "reg" not in per_task
