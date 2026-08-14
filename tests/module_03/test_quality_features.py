"""Tests for quality_features.py"""
import numpy as np
from module_03_sensor_fusion.quality_features import extract_quality_features
from module_03_sensor_fusion.config import QualityFeatureConfig


def test_extract_quality_features():
    T = 10
    validity_dict = {
        "outlier_masks": {"ch1": np.zeros(T, dtype=bool)},
        "missing_masks": {"ch1": np.zeros(T, dtype=bool)},
        "interpolated_masks": {"ch1": np.zeros(T, dtype=bool)},
    }
    validity_dict["outlier_masks"]["ch1"][3] = True
    validity_dict["missing_masks"]["ch1"][5] = True

    cfg = QualityFeatureConfig()
    mat, names = extract_quality_features(validity_dict, ["ch1"], cfg, num_timesteps=T)

    assert mat.shape == (T, 4)  # outlier, missing, interp, valid_flag
    assert "ch1_is_outlier" in names
    assert "ch1_is_valid" in names
    assert mat[3, names.index("ch1_is_outlier")] == 1.0
    assert mat[3, names.index("ch1_is_valid")] == 0.0
    assert mat[5, names.index("ch1_is_missing")] == 1.0
