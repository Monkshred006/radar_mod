"""Tests for BitNet Quantization-Aware Training (QAT)."""

import pytest
import torch
import tempfile
from torch.utils.data import DataLoader

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import RegressionHead
from module_05_training.config import TrainingConfig
from module_05_training.dataset import make_synthetic_scene_cache, PhotonShieldDataset, collate_module3
from module_05_training.target_adapter import SyntheticRegressionAdapter
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.layer_replacement import replace_linear_layers
from module_06_bitnet.qat import BitNetQATTrainer


class TestBitNetQAT:
    def test_qat_step_updates_master_weights(self, tmp_path):
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        engine = PhotonMambaHybrid(m_cfg)
        h_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=1)
        head = RegressionHead(32, h_cfg)

        b_cfg = BitNetConfig()
        replace_linear_layers(engine, b_cfg)
        replace_linear_layers(head, b_cfg)

        t_cfg = TrainingConfig(epochs=1, learning_rate=1e-3, batch_size=4, device="cpu")

        cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=30)
        adapter = SyntheticRegressionAdapter(num_outputs=1)
        ds = PhotonShieldDataset(cache, adapter, window_len=10, window_stride=5)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_module3)

        trainer = BitNetQATTrainer(engine, head, b_cfg, t_cfg, m_cfg)

        params_before = {n: p.clone().detach() for n, p in engine.named_parameters()}
        trainer.train_qat(loader, loader)

        changed = 0
        for n, p_before in params_before.items():
            p_after = dict(engine.named_parameters())[n]
            if not torch.allclose(p_before, p_after.detach()):
                changed += 1

        assert changed > 0, "No master weights were updated during QAT step"
