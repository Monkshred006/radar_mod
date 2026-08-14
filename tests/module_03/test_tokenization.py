"""Tests for tokenization.py"""
import numpy as np
import torch
from module_03_sensor_fusion.tokenization import SensorAwareTokenizer
from module_03_sensor_fusion.config import TokenizerConfig


def test_build_tokens_single():
    T = 10
    # Dummy fused matrix with 10 features: 0..3 optical, 3..5 env, 5..10 motion
    fused = np.random.randn(T, 10)
    gmap = {
        "optical": (0, 3),      # D=3
        "environment": (3, 5),  # D=2
        "motion": (5, 10),      # D=5
    }
    cfg = TokenizerConfig(explicit_group_order=["optical", "environment", "motion"])
    tok_layer = SensorAwareTokenizer(cfg)
    tokens, mask, groups = tok_layer.build_tokens_single(fused, gmap)

    assert groups == ["optical", "environment", "motion"]
    # Tokens shape should be [T, S, D_max] -> [10, 3, 5]
    assert tokens.shape == (10, 3, 5)
    assert mask.shape == (10, 3, 5)

    # Env feature dimension is 2, so index >= 2 should be padded (mask False)
    assert bool(mask[0, 1, 0]) is True
    assert bool(mask[0, 1, 1]) is True
    assert bool(mask[0, 1, 2]) is False


def test_build_tokens_batched():
    fused1 = np.random.randn(8, 6)
    fused2 = np.random.randn(8, 6)
    gmap = {"optical": (0, 2), "motion": (2, 6)}
    tok_layer = SensorAwareTokenizer()

    batched_tokens, batched_mask, groups = tok_layer.build_tokens_batched(
        [fused1, fused2], [gmap, gmap]
    )
    # Shape: [B, T, S, D_max] -> [2, 8, 2, 4]
    assert batched_tokens.shape == (2, 8, 2, 4)
    assert batched_mask.shape == (2, 8, 2, 4)
