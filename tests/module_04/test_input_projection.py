"""Unit tests for Input Projection module."""

import torch
import pytest
from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.input_projection import SensorTokenProjection


def test_sensor_token_projection_shapes():
    config = MambaHybridConfig(d_model=64, sensor_feature_dim=48, num_sensor_groups=5)
    proj = SensorTokenProjection(config)

    # Synthetic Module 3 output [B=2, T=10, S=5, D=48]
    tokens = torch.randn(2, 10, 5, 48)
    token_mask = torch.ones(2, 10, 5, 48, dtype=torch.bool)
    m3_output = {"tokens": tokens, "token_mask": token_mask}

    sensor_tokens, sensor_mask, aggregated = proj(m3_output)

    assert sensor_tokens.shape == (2, 10, 5, 64)
    assert sensor_mask.shape == (2, 10, 5)
    assert aggregated.shape == (2, 10, 64)


def test_sensor_token_projection_masking():
    config = MambaHybridConfig(d_model=64, sensor_feature_dim=48, num_sensor_groups=5)
    proj = SensorTokenProjection(config)

    tokens = torch.randn(1, 4, 5, 48)
    token_mask = torch.ones(1, 4, 5, 48, dtype=torch.bool)
    # Mask out sensor group 2 for all timesteps
    token_mask[0, :, 2, :] = False

    m3_output = {"tokens": tokens, "token_mask": token_mask}
    sensor_tokens, sensor_mask, aggregated = proj(m3_output)

    assert sensor_mask[0, 0, 2].item() is False
    # Padded sensor token should be zeroed out
    assert torch.allclose(sensor_tokens[0, :, 2, :], torch.tensor(0.0))
