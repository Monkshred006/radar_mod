"""Tests for fusion.py"""
import numpy as np
from module_03_sensor_fusion.fusion import fuse_sensor_features
from module_03_sensor_fusion.config import Module3Config


def test_fuse_sensor_features():
    T = 15
    ts = np.arange(T) * 0.1
    signals = {
        "photodiode_1": np.ones(T),
        "temperature": np.full(T, 25.0),
        "accel_x": np.zeros(T),
        "accel_y": np.zeros(T),
        "accel_z": np.zeros(T),
        "distance": np.full(T, 100.0),
    }
    validity = {
        "outlier_masks": {k: np.zeros(T, dtype=bool) for k in signals},
        "missing_masks": {k: np.zeros(T, dtype=bool) for k in signals},
        "interpolated_masks": {k: np.zeros(T, dtype=bool) for k in signals},
    }
    cfg = Module3Config()
    fused, names, gmap = fuse_sensor_features(signals, ts, validity, cfg)

    assert fused.shape[0] == T
    assert fused.shape[1] == len(names)
    assert "optical" in gmap
    assert "environment" in gmap
    assert "motion" in gmap
    assert "distance" in gmap
    assert "quality" in gmap

    # Indices in gmap should be sequential and valid
    for grp, (start, end) in gmap.items():
        assert 0 <= start < end <= fused.shape[1]
