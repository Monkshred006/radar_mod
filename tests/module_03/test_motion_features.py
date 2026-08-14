"""Tests for motion_features.py"""
import numpy as np
from module_03_sensor_fusion.motion_features import extract_motion_features
from module_03_sensor_fusion.config import MotionFeatureConfig


def test_extract_motion_features():
    T = 20
    ts = np.arange(T) * 0.05
    signals = {
        "accel_x": np.ones(T) * 1.0,
        "accel_y": np.ones(T) * 2.0,
        "accel_z": np.ones(T) * 2.0,
        "gyro_x": np.zeros(T),
        "gyro_y": np.zeros(T),
        "gyro_z": np.zeros(T),
    }
    cfg = MotionFeatureConfig()
    mat, names = extract_motion_features(signals, ts, list(signals.keys()), cfg)

    assert mat.shape[0] == T
    assert "accel_magnitude" in names
    assert "gyro_magnitude" in names

    # Accel mag should be sqrt(1^2 + 2^2 + 2^2) = 3.0
    accel_mag_idx = names.index("accel_magnitude")
    np.testing.assert_allclose(mat[:, accel_mag_idx], 3.0)
