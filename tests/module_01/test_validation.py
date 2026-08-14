"""Tests for validation module."""

import pytest
import numpy as np
from module_01_radar_input.validation import (
    validate_frame,
    validate_sequence,
    RadarDataValidationError,
    InconsistentDimensionError
)


def test_validate_frame_valid():
    arr = np.ones((64, 32), dtype=np.float32)
    validate_frame(arr)  # Should not raise exception
    validate_frame(arr, expected_shape=(64, 32))


def test_validate_frame_nan_inf():
    nan_arr = np.array([1.0, np.nan, 2.0])
    with pytest.raises(RadarDataValidationError, match="contains NaN"):
        validate_frame(nan_arr)

    inf_arr = np.array([1.0, np.inf, 2.0])
    with pytest.raises(RadarDataValidationError, match="contains Infinite"):
        validate_frame(inf_arr)


def test_validate_frame_shape_mismatch():
    arr = np.ones((64, 32))
    with pytest.raises(InconsistentDimensionError, match="Frame shape mismatch"):
        validate_frame(arr, expected_shape=(32, 64))


def test_validate_sequence_inconsistent_dim():
    f1 = np.ones((10, 10))
    f2 = np.ones((10, 12))
    with pytest.raises(InconsistentDimensionError, match="Inconsistent frame dimension"):
        validate_sequence([f1, f2])
