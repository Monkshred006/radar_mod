"""Tests for TargetAdapter and synthetic adapters."""

import pytest
import torch
import numpy as np

from module_05_training.target_adapter import (
    TargetAdapter,
    SyntheticRegressionAdapter,
    SyntheticClassificationAdapter,
    get_target_adapter,
)
from module_05_training.config import TrainingConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_sample(T=10, S=5, D=48, F=101):
    return {
        "tokens": torch.randn(T, S, D),
        "features": torch.randn(T, F),
        "timestamps": torch.linspace(0, 1, T),
    }


# ── TargetAdapter ─────────────────────────────────────────────────────────────

class TestTargetAdapter:
    def test_callable_regression(self):
        adapter = TargetAdapter(
            target_fn=lambda s: s["features"].mean(dim=0)[:3],
            target_type="regression",
            num_outputs=3,
        )
        sample = _make_sample()
        target = adapter(sample)
        assert isinstance(target, torch.Tensor)
        assert target.dtype == torch.float32
        assert target.shape == (3,)

    def test_callable_classification(self):
        adapter = TargetAdapter(
            target_fn=lambda s: torch.tensor(0),
            target_type="classification",
            num_classes=5,
        )
        sample = _make_sample()
        target = adapter(sample)
        assert target.dtype == torch.long
        assert target.shape == (1,)

    def test_numpy_input_converted(self):
        adapter = TargetAdapter(
            target_fn=lambda s: np.array([1.0, 2.0]),
            target_type="regression",
        )
        sample = _make_sample()
        target = adapter(sample)
        assert isinstance(target, torch.Tensor)
        assert target.dtype == torch.float32

    def test_scalar_input_gets_unsqueezed(self):
        adapter = TargetAdapter(
            target_fn=lambda s: 0.5,
            target_type="regression",
        )
        sample = _make_sample()
        target = adapter(sample)
        assert target.ndim >= 1

    def test_multitask_returns_dict(self):
        adapter = TargetAdapter(
            target_fn=lambda s: {"cls": torch.tensor(1), "reg": torch.tensor([0.5])},
            target_type="multitask",
        )
        sample = _make_sample()
        target = adapter(sample)
        assert isinstance(target, dict)
        assert "cls" in target and "reg" in target


# ── Synthetic adapters ────────────────────────────────────────────────────────

class TestSyntheticAdapters:
    def test_regression_shape(self):
        adapter = SyntheticRegressionAdapter(num_outputs=1)
        sample = _make_sample()
        t = adapter(sample)
        assert t.shape == (1,)
        assert t.dtype == torch.float32

    def test_regression_deterministic(self):
        adapter = SyntheticRegressionAdapter(num_outputs=2)
        sample = _make_sample()
        t1 = adapter(sample)
        t2 = adapter(sample)
        assert torch.allclose(t1, t2)

    def test_classification_dtype(self):
        adapter = SyntheticClassificationAdapter(num_classes=2)
        sample = _make_sample()
        t = adapter(sample)
        assert t.dtype == torch.long

    def test_get_target_adapter_regression(self):
        cfg = TrainingConfig(target_type="regression", num_regression_outputs=1)
        adapter = get_target_adapter(cfg)
        sample = _make_sample()
        t = adapter(sample)
        assert t.shape == (1,)

    def test_get_target_adapter_classification(self):
        cfg = TrainingConfig(target_type="classification", num_classes=3)
        adapter = get_target_adapter(cfg)
        sample = _make_sample()
        t = adapter(sample)
        assert t.dtype == torch.long
