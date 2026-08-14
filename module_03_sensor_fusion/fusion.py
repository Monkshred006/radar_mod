"""Deterministic sensor fusion module.

Combines group-specific features into a unified fused feature matrix [T, F_fused]
or batched tensor [B, T, F_fused].

No trainable weights or neural layers — pure deterministic fusion preserving
temporal ordering, channel/group identity, and quality information.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from module_03_sensor_fusion.config import Module3Config
from module_03_sensor_fusion.sensor_groups import SensorGroupRegistry
from module_03_sensor_fusion.optical_features import extract_optical_features
from module_03_sensor_fusion.environmental_features import extract_environmental_features
from module_03_sensor_fusion.motion_features import extract_motion_features
from module_03_sensor_fusion.distance_features import extract_distance_features
from module_03_sensor_fusion.quality_features import extract_quality_features


def fuse_sensor_features(
    signals: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    validity_dict: Dict[str, Dict[str, np.ndarray]],
    config: Module3Config,
    registry: Optional[SensorGroupRegistry] = None,
    causal: bool = True,
) -> Tuple[np.ndarray, List[str], Dict[str, Tuple[int, int]]]:
    """Extract and fuse all group features into a single matrix.

    Args:
        signals: Dict of channel_name -> np.ndarray [T]
        timestamps: np.ndarray [T]
        validity_dict: Validity masks dict from Module 2
        config: Module3Config
        registry: SensorGroupRegistry
        causal: Enforce causal feature extraction

    Returns:
        Tuple of:
        - fused_matrix: np.ndarray [T, F_total]
        - feature_names: List of all column names
        - group_index_map: Dict group_name -> (start_col, end_col)
    """
    reg = registry or SensorGroupRegistry()
    available_channels = set(signals.keys())
    T = len(timestamps)

    feature_blocks: List[np.ndarray] = []
    feature_names: List[str] = []
    group_map: Dict[str, Tuple[int, int]] = {}

    current_col = 0

    # 1. Optical Group
    opt_chs = reg.get_channels_in_group("optical", available_channels)
    if opt_chs and config.optical.enabled:
        mat, names = extract_optical_features(signals, timestamps, opt_chs, config.optical, causal=causal)
        if mat.shape[1] > 0:
            feature_blocks.append(mat)
            feature_names.extend(names)
            group_map["optical"] = (current_col, current_col + mat.shape[1])
            current_col += mat.shape[1]

    # 2. Environment Group
    env_chs = reg.get_channels_in_group("environment", available_channels)
    if env_chs and config.environmental.enabled:
        mat, names = extract_environmental_features(signals, timestamps, env_chs, config.environmental, causal=causal)
        if mat.shape[1] > 0:
            feature_blocks.append(mat)
            feature_names.extend(names)
            group_map["environment"] = (current_col, current_col + mat.shape[1])
            current_col += mat.shape[1]

    # 3. Motion Group
    mot_chs = reg.get_channels_in_group("motion", available_channels)
    if mot_chs and config.motion.enabled:
        mat, names = extract_motion_features(signals, timestamps, mot_chs, config.motion, causal=causal)
        if mat.shape[1] > 0:
            feature_blocks.append(mat)
            feature_names.extend(names)
            group_map["motion"] = (current_col, current_col + mat.shape[1])
            current_col += mat.shape[1]

    # 4. Distance Group
    dst_chs = reg.get_channels_in_group("distance", available_channels)
    if dst_chs and config.distance.enabled:
        mat, names = extract_distance_features(signals, timestamps, dst_chs, config.distance, causal=causal)
        if mat.shape[1] > 0:
            feature_blocks.append(mat)
            feature_names.extend(names)
            group_map["distance"] = (current_col, current_col + mat.shape[1])
            current_col += mat.shape[1]

    # 5. Quality Group
    if config.quality.enabled:
        all_chs = list(available_channels)
        mat, names = extract_quality_features(validity_dict, all_chs, config.quality, num_timesteps=T)
        if mat.shape[1] > 0:
            feature_blocks.append(mat)
            feature_names.extend(names)
            group_map["quality"] = (current_col, current_col + mat.shape[1])
            current_col += mat.shape[1]

    if feature_blocks:
        fused = np.column_stack(feature_blocks)
    else:
        fused = np.zeros((T, 0), dtype=np.float64)

    return fused, feature_names, group_map
