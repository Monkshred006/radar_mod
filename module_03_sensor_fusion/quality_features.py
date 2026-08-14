"""Quality and validity feature encoding module.

Converts Module 2 validity information (outlier, missing, interpolated masks)
into explicit model features per channel and timestep.

Allows the model to distinguish:
- "measured value zero" vs "sensor unavailable / missing"
- "valid measurement" vs "interpolated / outlier measurement"
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from module_03_sensor_fusion.config import QualityFeatureConfig


def extract_quality_features(
    validity_dict: Dict[str, Dict[str, np.ndarray]],
    channels: List[str],
    config: QualityFeatureConfig,
    num_timesteps: int,
) -> Tuple[np.ndarray, List[str]]:
    """Extract quality/validity features across all channels.

    Args:
        validity_dict: Dict from Module 2:
            {
                "outlier_masks": {ch: bool np.ndarray},
                "missing_masks": {ch: bool np.ndarray},
                "interpolated_masks": {ch: bool np.ndarray},
            }
        channels: List of sensor channels to encode quality features for
        config: QualityFeatureConfig
        num_timesteps: Length of sequence T

    Returns:
        Tuple of (feature_matrix [T, F], feature_names list[str])
    """
    if not config.enabled or not channels:
        return np.zeros((num_timesteps, 0), dtype=np.float64), []

    outlier_masks = validity_dict.get("outlier_masks", {})
    missing_masks = validity_dict.get("missing_masks", {})
    interp_masks = validity_dict.get("interpolated_masks", {})

    feature_cols: List[np.ndarray] = []
    feature_names: List[str] = []

    for ch in channels:
        o_mask = outlier_masks.get(ch, np.zeros(num_timesteps, dtype=bool)).astype(np.float64)
        m_mask = missing_masks.get(ch, np.zeros(num_timesteps, dtype=bool)).astype(np.float64)
        i_mask = interp_masks.get(ch, np.zeros(num_timesteps, dtype=bool)).astype(np.float64)

        if config.include_outlier_mask:
            feature_cols.append(o_mask)
            feature_names.append(f"{ch}_is_outlier")

        if config.include_missing_mask:
            feature_cols.append(m_mask)
            feature_names.append(f"{ch}_is_missing")

        if config.include_interpolated_mask:
            feature_cols.append(i_mask)
            feature_names.append(f"{ch}_is_interpolated")

        if config.include_valid_flag:
            # Valid flag is 1.0 if not missing and not outlier
            v_flag = ((m_mask == 0) & (o_mask == 0)).astype(np.float64)
            feature_cols.append(v_flag)
            feature_names.append(f"{ch}_is_valid")

    if feature_cols:
        mat = np.column_stack(feature_cols)
    else:
        mat = np.zeros((num_timesteps, 0), dtype=np.float64)

    return mat, feature_names
