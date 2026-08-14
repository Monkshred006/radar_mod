"""Tests for BitNet evaluation integration with Module 5."""

import pytest
import torch
from torch.utils.data import DataLoader

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import RegressionHead
from module_05_training.config import TrainingConfig
from module_05_training.dataset import make_synthetic_scene_cache, PhotonShieldDataset, collate_module3
from module_05_training.target_adapter import SyntheticRegressionAdapter
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.layer_replacement import replace_linear_layers
from module_06_bitnet.evaluation import evaluate_bitnet_model
from module_06_bitnet.pta_baseline import evaluate_ptq_baseline


class TestEvaluationIntegration:
    def test_evaluator_runs_on_bitnet(self):
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        engine = PhotonMambaHybrid(m_cfg)
        h_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=1)
        head = RegressionHead(32, h_cfg)

        b_cfg = BitNetConfig()
        replace_linear_layers(engine, b_cfg)
        replace_linear_layers(head, b_cfg)

        t_cfg = TrainingConfig(target_type="regression", num_regression_outputs=1, device="cpu")

        cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=30)
        adapter = SyntheticRegressionAdapter(num_outputs=1)
        ds = PhotonShieldDataset(cache, adapter, window_len=10, window_stride=5)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_module3)

        results = evaluate_bitnet_model(engine, head, t_cfg, loader)
        assert "loss" in results
        assert "metrics" in results
        assert torch.isfinite(torch.tensor(results["loss"]))

    def test_ptq_baseline_eval(self):
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        engine = PhotonMambaHybrid(m_cfg)
        h_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=1)
        head = RegressionHead(32, h_cfg)
        replace_linear_layers(engine, BitNetConfig())

        t_cfg = TrainingConfig(device="cpu")
        cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=30)
        ds = PhotonShieldDataset(cache, SyntheticRegressionAdapter(1), window_len=10, window_stride=5)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_module3)

        results = evaluate_ptq_baseline(engine, head, t_cfg, loader)
        assert results["model_variant"] == "Direct-Ternary PTQ"
