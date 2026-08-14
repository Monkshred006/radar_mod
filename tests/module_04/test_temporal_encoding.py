"""Unit tests for Temporal Encoding module."""

import torch
import pytest
from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.temporal_encoding import TemporalEncoding


def test_learned_temporal_encoding():
    config = MambaHybridConfig(d_model=64, use_temporal_encoding=True, temporal_encoding_type="learned")
    encoding = TemporalEncoding(config)

    x = torch.randn(2, 16, 64)
    out = encoding(x)

    assert out.shape == (2, 16, 64)
    # Output should differ from input due to positional embedding addition
    assert not torch.allclose(out, x)


def test_timestamp_delta_encoding():
    config = MambaHybridConfig(d_model=64, use_temporal_encoding=True, temporal_encoding_type="timestamp_delta")
    encoding = TemporalEncoding(config)

    x = torch.randn(2, 10, 64)
    timestamps = torch.linspace(0.0, 1.0, 10)  # [10]
    out = encoding(x, timestamps=timestamps)

    assert out.shape == (2, 10, 64)
    assert not torch.allclose(out, x)


def test_disabled_temporal_encoding():
    config = MambaHybridConfig(d_model=64, use_temporal_encoding=False)
    encoding = TemporalEncoding(config)

    x = torch.randn(2, 10, 64)
    out = encoding(x)

    assert torch.allclose(out, x)
