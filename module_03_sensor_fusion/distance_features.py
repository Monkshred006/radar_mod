"""Distance (VL53L0X) feature extraction module."""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from module_03_sensor_fusion.config import DistanceFeatureConfig
from module_03_sensor_fusion.feature_extractors import (
    compute_first_difference,
    compute_rate_of_change,
    compute_rolling_mean,
)


def extract_distance_features(
    signals: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    channels: List[str],
    config: DistanceFeatureConfig,
    causal: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Extract features for VL53L0X distance channels.

    Args:
        signals: Dict of channel_name -> np.ndarray [T]
        timestamps: np.ndarray [T]
        channels: List of distance channel names
        config: DistanceFeatureConfig
        causal: Enforce causal (streaming safe) extraction

    Returns:
        Tuple of (feature_matrix [T, F], feature_names list[str])
    """
    if not config.enabled or not channels:
        return np.zeros((len(timestamps), 0), dtype=np.float64), []

    T = len(timestamps)
    feature_cols: List[np.ndarray] = []
    feature_names: List[str] = []

    for ch in channels:
        val = signals.get(ch, np.zeros(T, dtype=np.float64))

        if config.current_val:
            feature_cols.append(val)
            feature_names.append(f"{ch}_value")

        if config.first_diff:
            diff = compute_first_difference(val, causal=causal)
            feature_cols.append(diff)
            feature_names.append(f"{ch}_first_diff")

        if config.rate_of_change:
            roc = compute_rate_of_change(val, timestamps, causal=causal)
            feature_cols.append(roc)
            feature_names.append(f"{ch}_rate_of_change")

        if config.rolling_mean:
            rmean = compute_rolling_mean(val, window=config.window_size, causal=causal)
            feature_cols.append(rmean)
            feature_names.append(f"{ch}_rolling_mean")

    if feature_cols:
        mat = np.column_stack(feature_cols)
    else:
        mat = np.zeros((T, 0), dtype=np.float64)

    return mat, feature_names
