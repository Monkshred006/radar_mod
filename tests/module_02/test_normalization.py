"""Tests for normalization.py — including anti-leakage verification."""
import json
import numpy as np
import pytest
from module_02_sensor_dsp.normalization import SensorScaler
from module_02_sensor_dsp.config import NormalizationConfig


def make_signal(seed=0, n=200, mean=5.0, std=2.0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


class TestMinMax:
    def test_fit_transform_range(self):
        sig = make_signal()
        scaler = SensorScaler("minmax")
        scaler.fit(sig)
        out = scaler.transform(sig)
        assert out.min() >= -1e-9
        assert out.max() <= 1.0 + 1e-9

    def test_inverse_roundtrip(self):
        sig = make_signal()
        scaler = SensorScaler("minmax")
        scaler.fit(sig)
        recovered = scaler.inverse_transform(scaler.transform(sig))
        np.testing.assert_allclose(recovered, sig, rtol=1e-5)

    def test_nan_preserved(self):
        sig = np.array([1.0, 2.0, np.nan, 4.0])
        scaler = SensorScaler("minmax")
        scaler.fit(sig)
        out = scaler.transform(sig)
        assert np.isnan(out[2])


class TestStandard:
    def test_mean_zero_std_one(self):
        sig = make_signal()
        scaler = SensorScaler("standard")
        scaler.fit(sig)
        out = scaler.transform(sig)
        assert abs(np.mean(out)) < 0.01
        assert abs(np.std(out) - 1.0) < 0.01

    def test_inverse_roundtrip(self):
        sig = make_signal(seed=1)
        scaler = SensorScaler("standard")
        scaler.fit(sig)
        recovered = scaler.inverse_transform(scaler.transform(sig))
        np.testing.assert_allclose(recovered, sig, rtol=1e-5)


class TestRobust:
    def test_output_not_nan(self):
        sig = make_signal()
        scaler = SensorScaler("robust")
        scaler.fit(sig)
        out = scaler.transform(sig)
        assert not np.isnan(out).any()

    def test_inverse_roundtrip(self):
        sig = make_signal(seed=2)
        scaler = SensorScaler("robust")
        scaler.fit(sig)
        recovered = scaler.inverse_transform(scaler.transform(sig))
        np.testing.assert_allclose(recovered, sig, rtol=1e-5)


class TestLeakagePrevention:
    """Verify training stats are never recalculated on val/test data."""

    def test_val_uses_train_stats(self):
        train_sig = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        val_sig = np.array([5.0, 6.0, 7.0], dtype=np.float64)

        scaler = SensorScaler("minmax")
        scaler.fit(train_sig)

        # Record stats after train fit
        train_min = scaler._min
        train_max = scaler._max

        # Transform val — must NOT refit
        out_val = scaler.transform(val_sig)

        assert scaler._min == train_min
        assert scaler._max == train_max
        # Val values > 1.0 because they exceed training max
        assert out_val[0] > 1.0

    def test_transform_without_fit_raises(self):
        scaler = SensorScaler("standard")
        with pytest.raises(RuntimeError, match="fit()"):
            scaler.transform(np.ones(5))


class TestStateSerialisation:
    def test_save_load_state(self, tmp_path):
        sig = make_signal()
        scaler = SensorScaler("standard")
        scaler.fit(sig)
        path = tmp_path / "scaler.json"
        scaler.save_state(path)
        loaded = SensorScaler.load_state(path)
        assert loaded._fitted
        np.testing.assert_allclose(
            scaler.transform(sig), loaded.transform(sig), rtol=1e-10
        )

    def test_json_contains_required_fields(self, tmp_path):
        sig = make_signal()
        scaler = SensorScaler("minmax")
        scaler.fit(sig)
        path = tmp_path / "s.json"
        scaler.save_state(path)
        with open(path) as f:
            state = json.load(f)
        for key in ("mean", "std", "min", "max", "method", "fitted"):
            assert key in state
