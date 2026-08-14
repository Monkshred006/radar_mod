"""Tests for filters.py"""
import numpy as np
import pytest
from module_02_sensor_dsp.filters import (
    apply_moving_average,
    apply_ema,
    apply_median_filter,
    apply_lowpass,
    apply_filter,
)
from module_02_sensor_dsp.config import FilterConfig


def make_sine(n=100, freq=1.0, rate=50.0):
    t = np.linspace(0, n / rate, n)
    return np.sin(2 * np.pi * freq * t)


class TestMovingAverage:
    def test_output_length(self):
        sig = np.ones(50)
        out = apply_moving_average(sig, window=5, causal=True)
        assert len(out) == len(sig)

    def test_constant_signal(self):
        sig = np.full(30, 3.0)
        out = apply_moving_average(sig, window=5, causal=True)
        np.testing.assert_allclose(out, 3.0, atol=1e-10)

    def test_causal_no_future(self):
        # Causal: output at t should not depend on samples after t
        sig = np.zeros(20)
        sig[10] = 100.0  # spike at t=10
        out_causal = apply_moving_average(sig, window=5, causal=True)
        # Before the spike, output must still be 0
        assert np.all(out_causal[:10] == 0.0)


class TestEMA:
    def test_output_length(self):
        sig = np.random.randn(80)
        out = apply_ema(sig, alpha=0.1)
        assert len(out) == 80

    def test_nan_preserved(self):
        sig = np.array([1.0, np.nan, 3.0])
        out = apply_ema(sig, alpha=0.5)
        assert np.isnan(out[1])

    def test_smoothing(self):
        noisy = np.random.randn(200)
        smooth = apply_ema(noisy, alpha=0.05)
        assert np.std(smooth) < np.std(noisy)


class TestMedianFilter:
    def test_output_length(self):
        sig = np.random.randn(50)
        out = apply_median_filter(sig, window=5, causal=True)
        assert len(out) == 50

    def test_spike_removal(self):
        sig = np.ones(30)
        sig[15] = 1000.0
        out = apply_median_filter(sig, window=5, causal=True)
        assert out[16] < 100.0  # spike dampened

    def test_causal(self):
        sig = np.zeros(20)
        sig[10] = 100.0
        out = apply_median_filter(sig, window=5, causal=True)
        assert np.all(out[:10] == 0.0)


class TestLowpass:
    def test_output_length(self):
        sig = make_sine(100)
        out = apply_lowpass(sig, cutoff_hz=5.0, sampling_rate_hz=50.0)
        assert len(out) == 100

    def test_attenuates_high_freq(self):
        t = np.linspace(0, 2, 200)
        low = np.sin(2 * np.pi * 1.0 * t)
        high = np.sin(2 * np.pi * 20.0 * t)
        sig = low + high
        out = apply_lowpass(sig, cutoff_hz=5.0, sampling_rate_hz=100.0, causal=False)
        # High-frequency component should be attenuated
        residual_std = np.std(out - low)
        assert residual_std < np.std(high) * 0.5


class TestApplyFilter:
    def test_dispatch_none(self):
        sig = np.array([1.0, 2.0, 3.0])
        cfg = FilterConfig(filter_type="none")
        out = apply_filter(sig, cfg)
        np.testing.assert_array_equal(out, sig)

    def test_invalid_type(self):
        sig = np.ones(10)
        cfg = FilterConfig(filter_type="invalid_type")  # type: ignore
        with pytest.raises(ValueError):
            apply_filter(sig, cfg)
