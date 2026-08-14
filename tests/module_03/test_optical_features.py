"""Tests for optical_features.py"""
import numpy as np
from module_03_sensor_fusion.optical_features import extract_optical_features
from module_03_sensor_fusion.config import OpticalFeatureConfig


def test_extract_optical_features():
    T = 20
    ts = np.arange(T) * 0.1
    signals = {
        "photodiode_1": np.sin(np.pi * ts),
        "photodiode_2": np.cos(np.pi * ts),
    }
    cfg = OpticalFeatureConfig()
    mat, names = extract_optical_features(signals, ts, ["photodiode_1", "photodiode_2"], cfg)

    assert mat.shape[0] == T
    assert mat.shape[1] == len(names)
    assert "photodiode_1_amplitude" in names
    assert "photodiode_1_first_diff" in names
    assert "photodiode_1_rolling_mean" in names


def test_optical_disabled():
    ts = np.arange(10) * 0.1
    signals = {"photodiode_1": np.ones(10)}
    cfg = OpticalFeatureConfig(enabled=False)
    mat, names = extract_optical_features(signals, ts, ["photodiode_1"], cfg)
    assert mat.shape == (10, 0)
    assert names == []
