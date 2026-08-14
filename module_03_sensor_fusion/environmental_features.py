"""Environmental (BME280) feature extraction module."""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from module_03_sensor_fusion.config import EnvironmentalFeatureConfig
from module_03_sensor_fusion.feature_extractors import (
    compute_first_difference,
    compute_rolling_mean,
)


def extract_environmental_features(
    signals: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    channels: List[str],
    config: EnvironmentalFeatureConfig,
    causal: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Extract features for BME280 environmental channels.

    Args:
        signals: Dict of channel_name -> np.ndarray [T]
        timestamps: np.ndarray [T]
        channels: List of environmental channel names (temp, humidity, pressure)
        config: EnvironmentalFeatureConfig
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

        if config.rolling_mean:
            rmean = compute_rolling_mean(val, window=config.window_size, causal=causal)
            feature_cols.append(rmean)
            feature_names.append(f"{ch}_rolling_mean")

    if feature_cols:
        mat = np.column_stack(feature_cols)
    else:
        mat = np.zeros((T, 0), dtype=np.float64)

    return mat, feature_names
