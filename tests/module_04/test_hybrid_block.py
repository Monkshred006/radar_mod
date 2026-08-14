"""Unit tests for Hybrid Block module."""

import torch
import pytest
from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.hybrid_block import HybridBlock


def test_hybrid_block_forward():
    config = MambaHybridConfig(d_model=64, backend="fallback")
    block = HybridBlock(config)

    x = torch.randn(2, 10, 64)
    sensor_tokens = torch.randn(2, 10, 5, 64)
    sensor_mask = torch.ones(2, 10, 5, dtype=torch.bool)

    out_x, out_tokens = block(x, sensor_tokens=sensor_tokens, sensor_mask=sensor_mask)

    assert out_x.shape == (2, 10, 64)
    assert out_tokens.shape == (2, 10, 5, 64)


def test_hybrid_block_gradients():
    config = MambaHybridConfig(d_model=64, backend="fallback")
    block = HybridBlock(config)

    x = torch.randn(2, 8, 64, requires_grad=True)
    out_x, _ = block(x)

    loss = out_x.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
