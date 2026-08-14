"""Tests for Module 5 metrics."""

import math
import pytest
import torch

from module_05_training.metrics import MetricsTracker, MultiTaskMetricsTracker


class TestRegressionMetrics:
    def setup_method(self):
        self.tracker = MetricsTracker(task_type="regression")

    def test_perfect_prediction(self):
        self.tracker.reset()
        t = torch.linspace(0, 1, 20).unsqueeze(1)
        self.tracker.update(t, t)
        m = self.tracker.compute()
        assert m["mae"] == pytest.approx(0.0, abs=1e-5)
        assert m["mse"] == pytest.approx(0.0, abs=1e-5)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-5)
        assert m["r2"] == pytest.approx(1.0, abs=1e-4)

    def test_constant_offset(self):
        self.tracker.reset()
        pred = torch.ones(10, 1)
        tgt = torch.zeros(10, 1)
        self.tracker.update(pred, tgt)
        m = self.tracker.compute()
        assert m["mae"] == pytest.approx(1.0, abs=1e-5)
        assert m["mse"] == pytest.approx(1.0, abs=1e-5)

    def test_accumulation_across_batches(self):
        self.tracker.reset()
        for _ in range(5):
            pred = torch.ones(4, 1) * 2.0
            tgt = torch.ones(4, 1) * 1.0
            self.tracker.update(pred, tgt)
        m = self.tracker.compute()
        assert m["mae"] == pytest.approx(1.0, abs=1e-5)

    def test_empty_returns_empty(self):
        self.tracker.reset()
        m = self.tracker.compute()
        assert m == {}


class TestClassificationMetrics:
    def setup_method(self):
        self.tracker = MetricsTracker(task_type="classification")

    def test_perfect_binary(self):
        self.tracker.reset()
        # logits: high positive = class 1
        logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])  # 2 samples, 2 classes
        targets = torch.tensor([0, 1])
        self.tracker.update(logits, targets)
        m = self.tracker.compute()
        assert m["accuracy"] == pytest.approx(1.0, abs=1e-5)

    def test_all_wrong(self):
        self.tracker.reset()
        logits = torch.tensor([[2.0, -2.0], [2.0, -2.0]])  # both predict class 0
        targets = torch.tensor([1, 1])
        self.tracker.update(logits, targets)
        m = self.tracker.compute()
        assert m["accuracy"] == pytest.approx(0.0, abs=1e-5)

    def test_f1_keys_present(self):
        self.tracker.reset()
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        self.tracker.update(logits, targets)
        m = self.tracker.compute()
        assert "f1_macro" in m
        assert "precision_macro" in m
        assert "recall_macro" in m


class TestMultiTaskMetrics:
    def test_per_task_metrics(self):
        tracker = MultiTaskMetricsTracker({
            "cls": "classification",
            "reg": "regression",
        })
        tracker.reset()
        cls_logits = torch.randn(8, 2)
        cls_targets = torch.randint(0, 2, (8,))
        reg_preds = torch.randn(8, 1)
        reg_targets = torch.randn(8, 1)
        tracker.update(
            {"cls": cls_logits, "reg": reg_preds},
            {"cls": cls_targets, "reg": reg_targets},
        )
        m = tracker.compute()
        assert any(k.startswith("cls/") for k in m)
        assert any(k.startswith("reg/") for k in m)
