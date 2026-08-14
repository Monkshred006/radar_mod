"""Tests for outliers.py"""
import numpy as np
import pytest
from module_02_sensor_dsp.outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_mad,
    detect_outliers_range,
    detect_outliers,
)
from module_02_sensor_dsp.config import OutlierConfig


def make_signal_with_spike():
    sig = np.ones(50) * 5.0
    sig[25] = 500.0  # clear outlier
    return sig


class TestIQR:
    def test_flags_spike(self):
        sig = make_signal_with_spike()
        mask = detect_outliers_iqr(sig, threshold=1.5)
        assert mask[25] is True or mask[25] == 1

    def test_no_false_positives_uniform(self):
        sig = np.ones(100) * 3.0
        mask = detect_outliers_iqr(sig, threshold=1.5)
        assert not mask.any()

    def test_nan_not_flagged(self):
        sig = np.array([1.0, np.nan, 1.0, 1.0, 100.0])
        mask = detect_outliers_iqr(sig, threshold=1.5)
        assert not mask[1]  # NaN should not be flagged


class TestZScore:
    def test_flags_spike(self):
        sig = make_signal_with_spike()
        mask = detect_outliers_zscore(sig, threshold=3.0)
        assert mask[25]

    def test_normal_data_no_flag(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(0, 1, 1000)
        mask = detect_outliers_zscore(sig, threshold=5.0)
        assert mask.sum() < 10  # at most a handful


class TestMAD:
    def test_flags_spike(self):
        sig = make_signal_with_spike()
        mask = detect_outliers_mad(sig, threshold=3.5)
        assert mask[25]

    def test_output_same_length(self):
        sig = np.random.randn(60)
        mask = detect_outliers_mad(sig)
        assert len(mask) == 60


class TestRange:
    def test_flags_below_min(self):
        sig = np.array([-10.0, 0.0, 50.0])
        mask = detect_outliers_range(sig, min_val=0.0, max_val=100.0)
        assert mask[0]
        assert not mask[1]
        assert not mask[2]

    def test_flags_above_max(self):
        sig = np.array([0.0, 50.0, 200.0])
        mask = detect_outliers_range(sig, min_val=0.0, max_val=100.0)
        assert mask[2]
        assert not mask[0]

    def test_nan_not_flagged(self):
        sig = np.array([np.nan, 50.0, 200.0])
        mask = detect_outliers_range(sig, min_val=0.0, max_val=100.0)
        assert not mask[0]


class TestDetectOutliers:
    def test_returns_tuple(self):
        sig = make_signal_with_spike()
        cfg = OutlierConfig(method="iqr", threshold=1.5)
        result_sig, mask = detect_outliers(sig, cfg)
        assert len(result_sig) == len(sig)
        assert len(mask) == len(sig)

    def test_none_method_no_flags(self):
        sig = np.array([1.0, 2.0, 100.0])
        cfg = OutlierConfig(method="none")
        _, mask = detect_outliers(sig, cfg)
        assert not mask.any()

    def test_values_not_modified(self):
        sig = make_signal_with_spike()
        cfg = OutlierConfig(method="zscore", threshold=3.0)
        result_sig, _ = detect_outliers(sig, cfg)
        # Original spike value must be preserved
        assert result_sig[25] == pytest.approx(500.0)
