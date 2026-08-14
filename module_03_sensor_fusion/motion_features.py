"""Motion (MPU6050) feature extraction module."""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from module_03_sensor_fusion.config import MotionFeatureConfig
from module_03_sensor_fusion.feature_extractors import (
    compute_first_difference,
    compute_rolling_mean,
    compute_rolling_std,
)


def extract_motion_features(
    signals: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    channels: List[str],
    config: MotionFeatureConfig,
    causal: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Extract features for MPU6050 accelerometer & gyroscope channels.

    Calculates:
    - Raw X/Y/Z channels
    - accel_magnitude = sqrt(ax^2 + ay^2 + az^2)
    - gyro_magnitude = sqrt(gx^2 + gy^2 + gz^2)
    - First temporal differences & rolling statistics

    Args:
        signals: Dict of channel_name -> np.ndarray [T]
        timestamps: np.ndarray [T]
        channels: List of motion channel names
        config: MotionFeatureConfig
        causal: Enforce causal (streaming safe) extraction

    Returns:
        Tuple of (feature_matrix [T, F], feature_names list[str])
    """
    if not config.enabled or not channels:
        return np.zeros((len(timestamps), 0), dtype=np.float64), []

    T = len(timestamps)
    feature_cols: List[np.ndarray] = []
    feature_names: List[str] = []

    # Separate accel and gyro channels
    accel_chs = [ch for ch in channels if "accel" in ch]
    gyro_chs = [ch for ch in channels if "gyro" in ch]

    # Raw channels & basic stats
    for ch in channels:
        val = signals.get(ch, np.zeros(T, dtype=np.float64))

        if config.raw_channels:
            feature_cols.append(val)
            feature_names.append(f"{ch}_value")

        if config.first_diff:
            diff = compute_first_difference(val, causal=causal)
            feature_cols.append(diff)
            feature_names.append(f"{ch}_first_diff")

        if config.rolling_mean:
            rmean = compute_rolling_mean(val, window=config.window_size, causal=causal)
            feature_cols.append(rmean)
            feature_names.append(f"{ch}_rolling_mean")

        if config.rolling_std:
            rstd = compute_rolling_std(val, window=config.window_size, causal=causal)
            feature_cols.append(rstd)
            feature_names.append(f"{ch}_rolling_std")

    # Accel magnitude: sqrt(ax^2 + ay^2 + az^2)
    if config.accel_magnitude and len(accel_chs) == 3:
        ax = signals.get("accel_x", np.zeros(T))
        ay = signals.get("accel_y", np.zeros(T))
        az = signals.get("accel_z", np.zeros(T))
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        feature_cols.append(accel_mag)
        feature_names.append("accel_magnitude")

        if config.first_diff:
            feature_cols.append(compute_first_difference(accel_mag, causal=causal))
            feature_names.append("accel_magnitude_first_diff")

    # Gyro magnitude: sqrt(gx^2 + gy^2 + gz^2)
    if config.gyro_magnitude and len(gyro_chs) == 3:
        gx = signals.get("gyro_x", np.zeros(T))
        gy = signals.get("gyro_y", np.zeros(T))
        gz = signals.get("gyro_z", np.zeros(T))
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        feature_cols.append(gyro_mag)
        feature_names.append("gyro_magnitude")

        if config.first_diff:
            feature_cols.append(compute_first_difference(gyro_mag, causal=causal))
            feature_names.append("gyro_magnitude_first_diff")

    if feature_cols:
        mat = np.column_stack(feature_cols)
    else:
        mat = np.zeros((T, 0), dtype=np.float64)

    return mat, feature_names
