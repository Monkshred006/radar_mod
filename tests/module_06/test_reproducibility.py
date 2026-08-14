"""Tests for reproducibility and FP32 reference integrity."""

import pytest
import torch
from module_05_training.reproducibility import set_seed
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.bit_linear import BitLinear


class TestReproducibility:
    def test_bit_linear_deterministic(self):
        set_seed(42)
        layer1 = BitLinear(16, 8)
        x1 = torch.randn(2, 16)
        out1 = layer1(x1)

        set_seed(42)
        layer2 = BitLinear(16, 8)
        x2 = torch.randn(2, 16)
        out2 = layer2(x2)

        assert torch.allclose(out1, out2)

    def test_bitnet_config_defaults(self):
        cfg = BitNetConfig()
        cfg.validate()
        assert cfg.enabled is True
        assert cfg.scaling_method == "mean_abs"
        assert cfg.scaling_scope == "per_tensor"
        assert cfg.activation_precision == "fp32"
        assert cfg.packing_enabled is False
