"""Tests for quality.py"""
import numpy as np
from module_02_sensor_dsp.quality import compute_channel_quality, compute_all_quality


def make_quality_inputs(n=100, missing_idx=None, outlier_idx=None, interp_idx=None):
    values = np.linspace(0.0, 1.0, n)
    timestamps = np.arange(n) * 0.1
    outlier_mask = np.zeros(n, dtype=bool)
    missing_mask = np.zeros(n, dtype=bool)
    interp_mask = np.zeros(n, dtype=bool)

    if missing_idx:
        for i in missing_idx:
            values[i] = np.nan
            missing_mask[i] = True
    if outlier_idx:
        for i in outlier_idx:
            outlier_mask[i] = True
    if interp_idx:
        for i in interp_idx:
            interp_mask[i] = True

    return values, outlier_mask, missing_mask, interp_mask, timestamps


class TestComputeChannelQuality:
    def test_returns_all_keys(self):
        vals, om, mm, im, ts = make_quality_inputs()
        q = compute_channel_quality(vals, om, mm, im, ts)
        expected_keys = {
            "n_samples", "missing_pct", "interpolated_pct", "outlier_pct",
            "nan_pct", "mean", "std", "variance", "signal_range",
            "saturation_pct", "timestamp_jitter_mean_s", "timestamp_jitter_max_s"
        }
        assert expected_keys.issubset(set(q.keys()))

    def test_missing_pct_correct(self):
        vals, om, mm, im, ts = make_quality_inputs(n=100, missing_idx=list(range(10)))
        q = compute_channel_quality(vals, om, mm, im, ts)
        assert abs(q["missing_pct"] - 10.0) < 0.1

    def test_outlier_pct_correct(self):
        vals, om, mm, im, ts = make_quality_inputs(n=100, outlier_idx=[0, 1, 2, 3, 4])
        q = compute_channel_quality(vals, om, mm, im, ts)
        assert abs(q["outlier_pct"] - 5.0) < 0.1

    def test_all_nan_signal(self):
        vals = np.full(50, np.nan)
        ts = np.arange(50) * 0.1
        om = mm = im = np.zeros(50, dtype=bool)
        q = compute_channel_quality(vals, om, mm, im, ts)
        assert np.isnan(q["mean"])
        assert q["nan_pct"] == 100.0

    def test_timestamp_jitter_uniform_grid(self):
        vals, om, mm, im, ts = make_quality_inputs()
        q = compute_channel_quality(vals, om, mm, im, ts)
        assert q["timestamp_jitter_mean_s"] < 1e-9


class TestComputeAllQuality:
    def test_multi_channel(self):
        n = 50
        ts = np.arange(n) * 0.1
        ch_vals = {
            "A": np.ones(n),
            "B": np.linspace(0, 5, n),
        }
        om = {"A": np.zeros(n, bool), "B": np.zeros(n, bool)}
        mm = {"A": np.zeros(n, bool), "B": np.zeros(n, bool)}
        im = {"A": np.zeros(n, bool), "B": np.zeros(n, bool)}
        q = compute_all_quality(ch_vals, om, mm, im, ts)
        assert "A" in q
        assert "B" in q
        assert q["A"]["mean"] == pytest.approx(1.0)


import pytest
