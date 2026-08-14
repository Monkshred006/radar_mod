"""Tests for selective layer replacement and inspection reporting."""

import pytest
import torch
import torch.nn as nn
from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import RegressionHead
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.bit_linear import BitLinear
from module_06_bitnet.layer_replacement import replace_linear_layers, generate_layer_inspection_report


class TestLayerReplacement:
    def test_selective_replacement_engine(self):
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        engine = PhotonMambaHybrid(m_cfg)

        cfg = BitNetConfig(
            quantize_input_projection=True,
            quantize_sensor_attention_qkv=True,
            quantize_sensor_attention_output=True,
            quantize_ffn=True,
            quantize_mamba_internal=False,
        )

        bit_engine, summary = replace_linear_layers(engine, cfg)
        assert summary["replaced_count"] > 0

        # Verify Mamba core is NOT converted to BitLinear
        for name, module in bit_engine.named_modules():
            if "mamba_branch" in name:
                assert not isinstance(module, BitLinear), f"Mamba core {name} was incorrectly converted!"

    def test_inspection_report(self):
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        engine = PhotonMambaHybrid(m_cfg)
        cfg = BitNetConfig()
        bit_engine, _ = replace_linear_layers(engine, cfg)

        report = generate_layer_inspection_report(bit_engine, cfg)
        assert "formatted_table" in report
        assert report["total_params"] > 0
        assert report["ternary_params"] > 0
