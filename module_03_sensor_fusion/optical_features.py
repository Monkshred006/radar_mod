"""Optical (BPW34 photodiode) feature extraction module."""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from module_03_sensor_fusion.config import OpticalFeatureConfig
from module_03_sensor_fusion.feature_extractors import (
    compute_first_difference,
    compute_rate_of_change,
    compute_rolling_mean,
    compute_rolling_std,
    compute_rolling_energy,
)


def extract_optical_features(
    signals: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    channels: List[str],
    config: OpticalFeatureConfig,
    causal: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Extract features for photodiode/optical channels.

    Args:
        signals: Dict of channel_name -> np.ndarray [T]
        timestamps: np.ndarray [T]
        channels: List of optical channel names
        config: OpticalFeatureConfig
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

        if config.amplitude:
            feature_cols.append(val)
            feature_names.append(f"{ch}_amplitude")

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

        if config.rolling_std:
            rstd = compute_rolling_std(val, window=config.window_size, causal=causal)
            feature_cols.append(rstd)
            feature_names.append(f"{ch}_rolling_std")

        if config.rolling_energy:
            reng = compute_rolling_energy(val, window=config.window_size, causal=causal)
            feature_cols.append(reng)
            feature_names.append(f"{ch}_rolling_energy")

    if feature_cols:
        mat = np.column_stack(feature_cols)
    else:
        mat = np.zeros((T, 0), dtype=np.float64)

    return mat, feature_names
