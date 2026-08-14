"""Unit tests for Cross-Sensor Interaction Branch."""

import torch
import pytest
from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.sensor_interaction import CrossSensorInteractionBranch


def test_cross_sensor_interaction_shapes():
    config = MambaHybridConfig(d_model=64, num_attention_heads=2)
    branch = CrossSensorInteractionBranch(config)

    # [B=2, T=10, S=5, D=64]
    sensor_tokens = torch.randn(2, 10, 5, 64)
    sensor_mask = torch.ones(2, 10, 5, dtype=torch.bool)

    interacted, agg = branch(sensor_tokens, sensor_mask=sensor_mask)

    assert interacted.shape == (2, 10, 5, 64)
    assert agg.shape == (2, 10, 64)


def test_cross_sensor_interaction_masking():
    config = MambaHybridConfig(d_model=64, num_attention_heads=2)
    branch = CrossSensorInteractionBranch(config)

    sensor_tokens = torch.randn(1, 4, 5, 64)
    sensor_mask = torch.ones(1, 4, 5, dtype=torch.bool)
    sensor_mask[0, :, 3:] = False  # Mask out last 2 sensor groups

    interacted, agg = branch(sensor_tokens, sensor_mask=sensor_mask)

    assert interacted.shape == (1, 4, 5, 64)
    assert agg.shape == (1, 4, 64)
    assert torch.isfinite(agg).all()
