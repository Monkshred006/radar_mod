"""Tests for environmental_features.py"""
import numpy as np
from module_03_sensor_fusion.environmental_features import extract_environmental_features
from module_03_sensor_fusion.config import EnvironmentalFeatureConfig


def test_extract_environmental_features():
    T = 15
    ts = np.arange(T) * 1.0
    signals = {
        "temperature": np.full(T, 25.0),
        "humidity": np.full(T, 50.0),
        "pressure": np.full(T, 1013.0),
    }
    cfg = EnvironmentalFeatureConfig()
    mat, names = extract_environmental_features(signals, ts, ["temperature", "humidity", "pressure"], cfg)

    assert mat.shape == (T, 9)  # 3 features * 3 channels
    assert "temperature_value" in names
    assert "temperature_first_diff" in names
    assert "temperature_rolling_mean" in names
