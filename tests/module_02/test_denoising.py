"""Tests for denoising.py"""
import numpy as np
import pytest
from module_02_sensor_dsp.denoising import (
    estimate_baseline_ema,
    estimate_baseline_percentile,
    remove_baseline,
    apply_baseline_correction,
)
from module_02_sensor_dsp.config import BaselineConfig


def test_baseline_ema_output_length():
    sig = np.random.randn(100) + 5.0
    baseline = estimate_baseline_ema(sig, alpha=0.01)
    assert len(baseline) == 100


def test_baseline_ema_tracks_dc():
    sig = np.full(200, 10.0)
    baseline = estimate_baseline_ema(sig, alpha=0.005)
    # After many samples the baseline should approach the dc level
    assert abs(baseline[-1] - 10.0) < 1.0


def test_baseline_percentile_output_length():
    sig = np.random.randn(80) + 3.0
    baseline = estimate_baseline_percentile(sig, window=20, percentile=5.0, causal=True)
    assert len(baseline) == 80


def test_remove_baseline_subtracts_correctly():
    sig = np.array([5.0, 6.0, 7.0])
    baseline = np.array([2.0, 2.0, 2.0])
    corrected = remove_baseline(sig, baseline)
    np.testing.assert_allclose(corrected, [3.0, 4.0, 5.0])


def test_apply_baseline_correction_disabled():
    sig = np.array([1.0, 2.0, 3.0])
    cfg = BaselineConfig(enabled=False)
    corrected, baseline = apply_baseline_correction(sig, cfg)
    np.testing.assert_array_equal(corrected, sig.astype(np.float64))
    np.testing.assert_array_equal(baseline, np.zeros(3))


def test_apply_baseline_correction_ema():
    # DC-offset signal: mean should shift toward zero after correction
    rng = np.random.default_rng(42)
    sig = rng.normal(0, 0.1, 200) + 5.0
    cfg = BaselineConfig(enabled=True, method="ema", alpha=0.005)
    corrected, baseline = apply_baseline_correction(sig, cfg)
    assert len(corrected) == 200
    # The corrected signal mean should be much closer to zero than raw
    assert abs(np.mean(corrected)) < abs(np.mean(sig))


def test_apply_baseline_correction_percentile():
    sig = np.abs(np.random.randn(100)) + 1.0
    cfg = BaselineConfig(enabled=True, method="percentile", window=20, percentile=5.0)
    corrected, baseline = apply_baseline_correction(sig, cfg)
    assert len(corrected) == 100


def test_invalid_baseline_method():
    sig = np.ones(10)
    cfg = BaselineConfig(enabled=True, method="unknown")  # type: ignore
    with pytest.raises(ValueError):
        apply_baseline_correction(sig, cfg)
