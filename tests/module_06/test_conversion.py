"""Tests for FP32 to BitNet model conversion pipeline."""

import pytest
import torch
import tempfile
from pathlib import Path

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import RegressionHead
from module_05_training.config import TrainingConfig
from module_05_training.checkpointing import save_checkpoint
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.model_conversion import convert_fp32_to_bitnet, compute_layer_conversion_stats


def _make_dummy_fp32_ckpt(tmp_path):
    m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
    t_cfg = TrainingConfig(target_type="regression", num_regression_outputs=1)
    engine = PhotonMambaHybrid(m_cfg)
    h_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=1)
    head = RegressionHead(32, h_cfg)

    combined = torch.nn.ModuleList([engine, head])
    opt = torch.optim.AdamW(combined.parameters())

    ckpt_path = tmp_path / "fp32_dummy.pt"
    save_checkpoint(
        path=str(ckpt_path),
        model=combined,
        optimizer=opt,
        scheduler=None,
        epoch=1,
        best_val_metric=0.5,
        training_config=t_cfg,
        model_config=m_cfg,
        history=[],
        seed=42,
    )
    return str(ckpt_path)


class TestModelConversion:
    def test_conversion_pipeline_runs(self, tmp_path):
        fp32_path = _make_dummy_fp32_ckpt(tmp_path)
        bitnet_path = str(tmp_path / "bitnet_converted.pt")
        b_cfg = BitNetConfig()

        engine_b, head_b, report = convert_fp32_to_bitnet(fp32_path, b_cfg, bitnet_path)

        assert Path(bitnet_path).exists()
        assert "layer_conversion_stats" in report
        assert "layer_inspection_report" in report

    def test_source_fp32_checkpoint_unmodified(self, tmp_path):
        fp32_path = _make_dummy_fp32_ckpt(tmp_path)
        bitnet_path = str(tmp_path / "bitnet_converted.pt")

        hash_before = Path(fp32_path).read_bytes()
        convert_fp32_to_bitnet(fp32_path, BitNetConfig(), bitnet_path)
        hash_after = Path(fp32_path).read_bytes()

        assert hash_before == hash_after, "FP32 source checkpoint was modified!"
