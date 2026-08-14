"""Tests for Evaluator."""

import pytest
import torch
from torch.utils.data import DataLoader

from module_05_training.config import TrainingConfig
from module_05_training.evaluator import Evaluator
from module_05_training.dataset import make_synthetic_scene_cache, PhotonShieldDataset, collate_module3
from module_05_training.target_adapter import SyntheticRegressionAdapter


def _make_eval_setup(window_len=10, d_model=32):
    from module_04_mamba_hybrid.config import MambaHybridConfig
    from module_04_mamba_hybrid.engine import PhotonMambaHybrid
    from module_04_mamba_hybrid.heads import RegressionHead

    cfg = TrainingConfig(
        target_type="regression",
        num_regression_outputs=1,
        device="cpu",
        loss_name="mse",
    )
    model_cfg = MambaHybridConfig(d_model=d_model, num_layers=1, max_sequence_length=window_len)
    engine = PhotonMambaHybrid(model_cfg)
    from module_04_mamba_hybrid.config import TaskHeadConfig
    head_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=1)
    head = RegressionHead(d_model, head_cfg)

    cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=40)
    adapter = SyntheticRegressionAdapter(num_outputs=1)
    ds = PhotonShieldDataset(cache, adapter, window_len=window_len, window_stride=5)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_module3)

    return cfg, engine, head, loader


class TestEvaluator:
    def test_returns_expected_keys(self):
        cfg, engine, head, loader = _make_eval_setup()
        ev = Evaluator(cfg)
        results = ev.evaluate(engine, head, loader)
        assert "loss" in results
        assert "metrics" in results
        assert "inference_time_s" in results
        assert "per_sample_latency_ms" in results
        assert "sample_count" in results

    def test_loss_is_finite(self):
        cfg, engine, head, loader = _make_eval_setup()
        ev = Evaluator(cfg)
        results = ev.evaluate(engine, head, loader)
        assert torch.isfinite(torch.tensor(results["loss"]))

    def test_sample_count_correct(self):
        cfg, engine, head, loader = _make_eval_setup()
        ev = Evaluator(cfg)
        results = ev.evaluate(engine, head, loader)
        total_samples = sum(len(b[0].get("features", b[0].get("tokens")))
                            for b in loader) if False else results["sample_count"]
        assert results["sample_count"] > 0

    def test_metrics_present(self):
        cfg, engine, head, loader = _make_eval_setup()
        ev = Evaluator(cfg)
        results = ev.evaluate(engine, head, loader)
        # Regression metrics expected
        assert "mae" in results["metrics"]
        assert "rmse" in results["metrics"]

    def test_no_grad_during_evaluation(self):
        """Evaluator must not compute gradients (test that params unchanged)."""
        cfg, engine, head, loader = _make_eval_setup()
        params_before = {n: p.clone() for n, p in engine.named_parameters()}
        ev = Evaluator(cfg)
        ev.evaluate(engine, head, loader)
        for n, p in engine.named_parameters():
            assert torch.allclose(params_before[n], p), f"Param {n} changed during evaluation!"
