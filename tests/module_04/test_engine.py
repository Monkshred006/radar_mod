"""Unit tests for PhotonMambaHybrid Engine."""

import torch
import pytest
from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid


def test_engine_forward_with_tokens():
    config = MambaHybridConfig(d_model=64, num_layers=2, backend="fallback")
    engine = PhotonMambaHybrid(config)

    tokens = torch.randn(2, 12, 5, 48)
    token_mask = torch.ones(2, 12, 5, 48, dtype=torch.bool)
    timestamps = torch.linspace(0, 1, 12)
    m3_out = {"tokens": tokens, "token_mask": token_mask, "timestamps": timestamps}

    res = engine(m3_out)

    assert "sequence_output" in res
    assert "pooled_output" in res
    assert res["sequence_output"].shape == (2, 12, 64)
    assert res["pooled_output"].shape == (2, 64)


def test_engine_ablation_modes():
    # Test Mamba-only, Interaction-only, and Hybrid configurations
    x_tokens = torch.randn(1, 8, 5, 48)
    m3_out = {"tokens": x_tokens}

    for use_mamba, use_attn in [(True, False), (False, True), (True, True)]:
        cfg = MambaHybridConfig(
            d_model=32,
            num_layers=1,
            use_mamba=use_mamba,
            use_sensor_attention=use_attn,
            backend="fallback",
        )
        eng = PhotonMambaHybrid(cfg)
        out = eng(m3_out)
        assert out["pooled_output"].shape == (1, 32)
