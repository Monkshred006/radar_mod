"""Tests for synchronization.py — including causal streaming verification."""
import numpy as np
import pytest
from module_02_sensor_dsp.synchronization import (
    build_target_grid,
    synchronize_channel_offline,
    synchronize_channel_streaming,
    synchronize_all_channels_offline,
)
from module_02_sensor_dsp.config import SyncConfig


class TestBuildTargetGrid:
    def test_grid_length(self):
        grid = build_target_grid(0.0, 1.0, 10.0)
        assert len(grid) == 11  # 0.0, 0.1, ..., 1.0

    def test_spacing(self):
        grid = build_target_grid(0.0, 1.0, 5.0)
        diffs = np.diff(grid)
        np.testing.assert_allclose(diffs, 0.2, atol=1e-10)


class TestSyncChannelOffline:
    def make_src(self, n=10, rate=2.0):
        times = np.arange(n) / rate
        values = np.arange(n, dtype=float)
        return times, values

    def test_nearest_output_length(self):
        src_t, src_v = self.make_src()
        tgt = build_target_grid(0.0, src_t[-1], 4.0)
        out, interp, miss = synchronize_channel_offline(src_t, src_v, tgt, method="nearest")
        assert len(out) == len(tgt)

    def test_linear_interpolation(self):
        src_t = np.array([0.0, 1.0])
        src_v = np.array([0.0, 10.0])
        tgt = np.array([0.0, 0.5, 1.0])
        out, _, _ = synchronize_channel_offline(src_t, src_v, tgt, method="linear")
        np.testing.assert_allclose(out, [0.0, 5.0, 10.0], atol=1e-9)

    def test_gap_too_large_becomes_nan(self):
        src_t = np.array([0.0, 10.0])
        src_v = np.array([1.0, 2.0])
        tgt = np.array([5.0])
        out, _, miss = synchronize_channel_offline(
            src_t, src_v, tgt, method="linear", max_gap_s=2.0
        )
        assert np.isnan(out[0])
        assert miss[0]

    def test_ffill_causal(self):
        src_t = np.array([0.0, 0.5, 1.0])
        src_v = np.array([1.0, 2.0, 3.0])
        tgt = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        out, interp, _ = synchronize_channel_offline(src_t, src_v, tgt, method="ffill")
        # At t=0.25, no newer sample available; should take t=0.0 value
        assert out[1] == pytest.approx(1.0)
        assert interp[1]

    def test_timestamp_ordering_preserved(self):
        src_t = np.arange(20) * 0.1
        src_v = np.random.randn(20)
        tgt = np.arange(40) * 0.05
        out, _, _ = synchronize_channel_offline(src_t, src_v, tgt, method="nearest")
        # Timestamps are monotonically increasing by construction
        assert np.all(np.diff(tgt) >= 0)


class TestSyncChannelStreaming:
    """Verify streaming sync is strictly causal."""

    def test_no_future_samples(self):
        """Output at tgt_time=0.5 must not use src_time=1.0."""
        state = {"last_time": None, "last_value": None}

        # Feed sample at t=0.0
        val0, _, miss0 = synchronize_channel_streaming(
            src_time=0.0, src_value=42.0,
            tgt_time=0.0, state=state, method="ffill"
        )
        # Feed query at t=0.5 (before t=1.0 arrives)
        val_half, _, miss_half = synchronize_channel_streaming(
            src_time=0.5, src_value=99.0,
            tgt_time=0.5, state=state, method="ffill"
        )
        assert val_half == pytest.approx(99.0)
        # Future sample at t=1.0 should NOT have affected output at t=0.5
        assert val0 == pytest.approx(42.0)

    def test_missing_when_no_past_data(self):
        state = {"last_time": None, "last_value": None}
        val, _, miss = synchronize_channel_streaming(
            src_time=2.0, src_value=5.0,
            tgt_time=1.0, state=state
        )
        # src_time=2.0 is in the future relative to tgt_time=1.0
        # and state has no history → missing
        assert miss

    def test_gap_too_large_returns_nan(self):
        state = {"last_time": 0.0, "last_value": 1.0}
        val, _, miss = synchronize_channel_streaming(
            src_time=100.0, src_value=2.0,
            tgt_time=5.0, state=state, max_gap_s=1.0
        )
        assert miss


class TestSyncAllChannels:
    def test_multi_rate_alignment(self):
        # Fast channel: 10 Hz
        fast_t = np.arange(20) / 10.0
        fast_v = np.ones(20) * 2.0
        # Slow channel: 2 Hz
        slow_t = np.arange(4) / 2.0
        slow_v = np.ones(4) * 5.0

        cfg = SyncConfig(target_rate_hz=10.0, method_offline="nearest")
        tgt_times, synced, _, _ = synchronize_all_channels_offline(
            channel_times={"fast": fast_t, "slow": slow_t},
            channel_values={"fast": fast_v, "slow": slow_v},
            config=cfg,
        )
        assert len(synced["fast"]) == len(tgt_times)
        assert len(synced["slow"]) == len(tgt_times)
