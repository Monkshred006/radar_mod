"""Tests for BitLinear layer."""

import pytest
import torch
import torch.nn as nn
from module_06_bitnet.bit_linear import BitLinear
from module_06_bitnet.config import BitNetConfig


class TestBitLinear:
    def test_forward_pass_output_shape(self):
        layer = BitLinear(in_features=16, out_features=8)
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 8)
        assert torch.isfinite(out).all()

    def test_backward_pass_gradients(self):
        layer = BitLinear(in_features=8, out_features=4)
        x = torch.randn(2, 8, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    def test_from_linear_copies_parameters(self):
        linear = nn.Linear(10, 5)
        bit_layer = BitLinear.from_linear(linear)
        assert bit_layer.in_features == 10
        assert bit_layer.out_features == 5
        assert torch.equal(bit_layer.weight.data, linear.weight.data)
        assert torch.equal(bit_layer.bias.data, linear.bias.data)

    def test_disabled_quantization_equals_linear(self):
        cfg = BitNetConfig(enabled=False)
        linear = nn.Linear(8, 4)
        bit_layer = BitLinear.from_linear(linear, config=cfg)

        x = torch.randn(2, 8)
        out_linear = linear(x)
        out_bit = bit_layer(x)
        assert torch.allclose(out_linear, out_bit)
