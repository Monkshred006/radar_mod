"""Tests for BitNet checkpointing."""

import pytest
import torch
import tempfile
from pathlib import Path

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import RegressionHead
from module_05_training.config import TrainingConfig
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.layer_replacement import replace_linear_layers
from module_06_bitnet.checkpointing import save_bitnet_checkpoint, load_bitnet_checkpoint


class TestBitNetCheckpointing:
    def test_save_load_roundtrip(self, tmp_path):
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        engine = PhotonMambaHybrid(m_cfg)
        h_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=1)
        head = RegressionHead(32, h_cfg)

        b_cfg = BitNetConfig(scaling_method="max_abs")
        replace_linear_layers(engine, b_cfg)
        replace_linear_layers(head, b_cfg)

        t_cfg = TrainingConfig()
        ckpt_path = str(tmp_path / "bitnet_ckpt.pt")

        save_bitnet_checkpoint(
            path=ckpt_path,
            engine=engine,
            head=head,
            bitnet_config=b_cfg,
            model_config=m_cfg,
            training_config=t_cfg,
            source_fp32_checkpoint="dummy_source.pt",
            epoch=3,
        )
        assert Path(ckpt_path).exists()

        # Build fresh model and load
        engine2 = PhotonMambaHybrid(m_cfg)
        head2 = RegressionHead(32, h_cfg)
        replace_linear_layers(engine2, b_cfg)
        replace_linear_layers(head2, b_cfg)

        raw = load_bitnet_checkpoint(ckpt_path, engine2, head2)

        assert raw["epoch"] == 3
        assert raw["source_fp32_checkpoint"] == "dummy_source.pt"

        # Parameters must match
        for (n1, p1), (n2, p2) in zip(engine.named_parameters(), engine2.named_parameters()):
            assert torch.allclose(p1, p2)
