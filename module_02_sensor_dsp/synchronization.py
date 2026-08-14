"""Timestamp synchronization and multi-rate sensor alignment.

Aligns multiple sensor channels (potentially at different sampling rates)
onto a uniform target temporal grid.

OFFLINE mode: supports nearest, forward-fill, and linear interpolation.
STREAMING mode: ONLY causal methods — nearest or forward-fill (zero-order hold).
                Future samples are never used.

Missing/interpolated values are tracked in a quality metadata structure.
Gaps exceeding max_gap_s result in NaN values (flagged as missing, not filled).
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from module_02_sensor_dsp.config import SyncConfig


def build_target_grid(
    t_start: float,
    t_end: float,
    rate_hz: float,
) -> np.ndarray:
    """Create a uniform timestamp grid.

    Args:
        t_start: Start time (seconds).
        t_end: End time (seconds).
        rate_hz: Target sampling rate in Hz.

    Returns:
        1-D float64 array of evenly-spaced timestamps.
    """
    dt = 1.0 / rate_hz
    return np.arange(t_start, t_end + dt * 0.5, dt, dtype=np.float64)


def _find_nearest_indices(
    src_times: np.ndarray,
    tgt_times: np.ndarray,
) -> np.ndarray:
    """For each target time find the index of the nearest source time."""
    indices = np.searchsorted(src_times, tgt_times, side="left")
    left = np.clip(indices - 1, 0, len(src_times) - 1)
    right = np.clip(indices, 0, len(src_times) - 1)
    use_right = np.abs(tgt_times - src_times[right]) <= np.abs(tgt_times - src_times[left])
    result = np.where(use_right, right, left)
    return result


def synchronize_channel_offline(
    src_times: np.ndarray,
    src_values: np.ndarray,
    tgt_times: np.ndarray,
    method: str = "linear",
    max_gap_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a single channel onto the target timestamp grid (offline).

    Args:
        src_times: Source timestamps (1-D, sorted ascending).
        src_values: Source values (1-D, same length as src_times).
        tgt_times: Target grid timestamps.
        method: "nearest", "ffill", or "linear".
        max_gap_s: Max gap in seconds before marking as missing (NaN).

    Returns:
        Tuple of:
        - resampled_values: float64 array aligned to tgt_times
        - interpolated_mask: bool array (True = was interpolated/filled)
        - missing_mask: bool array (True = gap too large, set to NaN)
    """
    src_times = np.asarray(src_times, dtype=np.float64)
    src_values = np.asarray(src_values, dtype=np.float64)
    n_tgt = len(tgt_times)
    out = np.full(n_tgt, np.nan, dtype=np.float64)
    interpolated = np.zeros(n_tgt, dtype=bool)
    missing = np.zeros(n_tgt, dtype=bool)

    if len(src_times) == 0:
        missing[:] = True
        return out, interpolated, missing

    if method == "nearest":
        idx = _find_nearest_indices(src_times, tgt_times)
        for i, (t, j) in enumerate(zip(tgt_times, idx)):
            gap = abs(t - src_times[j])
            if gap > max_gap_s:
                missing[i] = True
            else:
                out[i] = src_values[j]
                if gap > 0:
                    interpolated[i] = True

    elif method == "ffill":
        j = 0
        for i, t in enumerate(tgt_times):
            # advance source pointer
            while j < len(src_times) - 1 and src_times[j + 1] <= t:
                j += 1
            if src_times[j] > t:
                missing[i] = True
            else:
                gap = t - src_times[j]
                if gap > max_gap_s:
                    missing[i] = True
                else:
                    out[i] = src_values[j]
                    if gap > 0:
                        interpolated[i] = True

    elif method == "linear":
        for i, t in enumerate(tgt_times):
            idx_r = np.searchsorted(src_times, t, side="right")
            idx_l = idx_r - 1
            if idx_l < 0:
                # before first sample
                gap = src_times[0] - t
                if gap <= max_gap_s:
                    out[i] = src_values[0]
                    interpolated[i] = True
                else:
                    missing[i] = True
            elif idx_r >= len(src_times):
                # after last sample
                gap = t - src_times[-1]
                if gap <= max_gap_s:
                    out[i] = src_values[-1]
                    interpolated[i] = True
                else:
                    missing[i] = True
            else:
                t0, t1 = src_times[idx_l], src_times[idx_r]
                v0, v1 = src_values[idx_l], src_values[idx_r]
                gap = t1 - t0
                if gap > max_gap_s:
                    missing[i] = True
                else:
                    alpha = (t - t0) / (t1 - t0)
                    out[i] = v0 + alpha * (v1 - v0)
                    if alpha > 0:
                        interpolated[i] = True
    else:
        raise ValueError(f"Unknown sync method: {method!r}")

    return out, interpolated, missing


def synchronize_channel_streaming(
    src_time: float,
    src_value: float,
    tgt_time: float,
    state: Dict,  # mutable state dict: {"last_time": float, "last_value": float}
    method: str = "ffill",
    max_gap_s: float = 1.0,
) -> Tuple[float, bool, bool]:
    """Resample one sample for a channel in streaming (causal) mode.

    Only forward-fill and nearest are allowed (no future samples).

    Args:
        src_time: Incoming source sample timestamp.
        src_value: Incoming source sample value.
        tgt_time: Target grid timestamp to produce output for.
        state: Mutable dict holding last seen value (updated in place).
        method: "ffill" or "nearest".
        max_gap_s: Max allowable gap before returning NaN.

    Returns:
        Tuple of (resampled_value, was_interpolated, is_missing).
    """
    # Update state with latest source sample if it's at or before target time
    if src_time <= tgt_time and not np.isnan(src_value):
        state["last_time"] = src_time
        state["last_value"] = src_value

    last_t = state.get("last_time", None)
    last_v = state.get("last_value", None)

    if last_t is None or last_v is None:
        return np.nan, False, True  # no past data yet

    gap = tgt_time - last_t
    if gap > max_gap_s:
        return np.nan, False, True

    return float(last_v), (gap > 0), False


def synchronize_all_channels_offline(
    channel_times: Dict[str, np.ndarray],
    channel_values: Dict[str, np.ndarray],
    config: SyncConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Align all sensor channels onto a common target grid (offline).

    Args:
        channel_times: Map from channel name to timestamps array.
        channel_values: Map from channel name to values array.
        config: SyncConfig with target rate and method.

    Returns:
        Tuple of:
        - tgt_times: Unified timestamp grid (float64)
        - synced_values: Dict channel -> resampled values
        - interpolated_masks: Dict channel -> interpolated boolean array
        - missing_masks: Dict channel -> missing boolean array
    """
    all_times = [t for t in channel_times.values() if len(t) > 0]
    if not all_times:
        raise ValueError("No channel data provided for synchronization.")

    t_start = float(min(t.min() for t in all_times))
    t_end = float(max(t.max() for t in all_times))
    tgt_times = build_target_grid(t_start, t_end, config.target_rate_hz)

    synced: Dict[str, np.ndarray] = {}
    interpolated: Dict[str, np.ndarray] = {}
    missing: Dict[str, np.ndarray] = {}

    for ch_name in channel_times:
        s_vals, interp, miss = synchronize_channel_offline(
            src_times=channel_times[ch_name],
            src_values=channel_values[ch_name],
            tgt_times=tgt_times,
            method=config.method_offline,
            max_gap_s=config.max_gap_s,
        )
        synced[ch_name] = s_vals
        interpolated[ch_name] = interp
        missing[ch_name] = miss

    return tgt_times, synced, interpolated, missing
