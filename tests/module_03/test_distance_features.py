"""Tests for distance_features.py"""
import numpy as np
from module_03_sensor_fusion.distance_features import extract_distance_features
from module_03_sensor_fusion.config import DistanceFeatureConfig


def test_extract_distance_features():
    T = 10
    ts = np.arange(T) * 0.1
    signals = {"distance": np.linspace(100.0, 200.0, T)}
    cfg = DistanceFeatureConfig()
    mat, names = extract_distance_features(signals, ts, ["distance"], cfg)

    assert mat.shape == (T, 4)
    assert "distance_value" in names
    assert "distance_rate_of_change" in names
