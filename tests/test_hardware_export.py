"""Unit tests for Hardware-Aware Export Pipeline (ONNX, INT8, Ternary Packing, Uno Q Profiler)."""

import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_06_bitnet.export_onnx import export_photon_v0_onnx
from module_06_bitnet.quantize_int8 import quantize_photon_v0_int8
from module_06_bitnet.pack_ternary import (
    quantize_to_ternary,
    pack_ternary_matrix_to_uint8,
    pack_model_ternary,
)
from module_06_bitnet.profile_uno_q import (
    estimate_photon_v0_macs,
    profile_for_uno_q,
)


class TestHardwareExport:
    """Test suite for hardware export and profiling modules."""

    def test_estimate_macs(self):
        macs = estimate_photon_v0_macs(
            input_dim=64,
            hidden_dim=64,
            num_layers=2,
            sequence_length=16,
            num_classes=4,
        )
        assert macs > 10_000
        assert isinstance(macs, int)

    def test_profile_uno_q(self):
        model = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16)
        profile = profile_for_uno_q(model=model, sequence_length=16)
        assert "parameter_count" in profile
        assert "weights_int8_kb" in profile
        assert "peak_sram_int8_kb" in profile
        assert "overall_fit" in profile
        assert profile["parameter_count"] > 0
        assert profile["weights_int8_kb"] < profile["target_flash_kb"]

    def test_quantize_int8(self):
        model = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "quant_int8.pt"
            q_model, stats = quantize_photon_v0_int8(model, output_path=out_file)
            assert out_file.exists()
            assert stats["reduction_factor"] > 1.0
            assert stats["int8_size_bytes"] < stats["fp32_size_bytes"]

    def test_quantize_and_pack_ternary(self):
        # 1. Test single matrix quantization
        w = torch.randn(32, 64)
        ternary, alpha = quantize_to_ternary(w)
        unique_vals = set(torch.unique(ternary).tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})
        assert alpha > 0.0

        # 2. Test packing 4 weights per uint8 byte
        packed = pack_ternary_matrix_to_uint8(ternary.numpy())
        assert packed.dtype == np.uint8
        assert len(packed) == (32 * 64) // 4

        # 3. Test full model ternary packing
        model = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_npz = Path(tmpdir) / "ternary_packed.npz"
            pack_stats = pack_model_ternary(model, output_path=out_npz)
            assert out_npz.exists()
            assert pack_stats["compression_ratio"] > 10.0  # ~16x compression

    def test_export_onnx(self):
        model = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16, backend="fallback")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_onnx = Path(tmpdir) / "model.onnx"
            res = export_photon_v0_onnx(model=model, output_path=out_onnx, sequence_length=16)
            assert out_onnx.exists()
            assert out_onnx.stat().st_size > 1000
