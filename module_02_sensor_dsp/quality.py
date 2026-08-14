"""Signal quality metrics.

Computes per-channel quality statistics returned alongside processed signals.
These metrics inform downstream modules (Module 3) about data reliability.

No quality thresholds are hard-coded; all decisions are deferred to consumers.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np


def compute_channel_quality(
    values: np.ndarray,
    outlier_mask: np.ndarray,
    missing_mask: np.ndarray,
    interpolated_mask: np.ndarray,
    timestamps: np.ndarray,
) -> Dict[str, Any]:
    """Compute quality metrics for a single preprocessed channel.

    Args:
        values: Processed signal values (float64, may contain NaN).
        outlier_mask: Boolean array (True = outlier flagged).
        missing_mask: Boolean array (True = missing/gap-too-large).
        interpolated_mask: Boolean array (True = interpolated/filled).
        timestamps: Aligned timestamps for this channel.

    Returns:
        Dictionary of quality metrics:
        - n_samples: total number of samples
        - missing_pct: percentage of missing samples
        - interpolated_pct: percentage of interpolated/filled samples
        - outlier_pct: percentage of outlier-flagged samples
        - nan_pct: percentage of NaN values in processed output
        - mean: signal mean (excluding NaN)
        - std: signal standard deviation (excluding NaN)
        - signal_range: [min, max] of valid values
        - variance: variance of valid values
        - saturation_pct: percentage of samples at/near min or max
        - timestamp_jitter_mean_s: mean abs deviation from expected spacing
        - timestamp_jitter_max_s: max abs deviation from expected spacing
    """
    n = len(values)
    valid_vals = values[~np.isnan(values)]

    missing_pct = 100.0 * np.sum(missing_mask) / n if n > 0 else 0.0
    interp_pct = 100.0 * np.sum(interpolated_mask) / n if n > 0 else 0.0
    outlier_pct = 100.0 * np.sum(outlier_mask) / n if n > 0 else 0.0
    nan_pct = 100.0 * np.sum(np.isnan(values)) / n if n > 0 else 0.0

    if len(valid_vals) > 0:
        mean_val = float(np.mean(valid_vals))
        std_val = float(np.std(valid_vals))
        var_val = float(np.var(valid_vals))
        sig_min = float(np.min(valid_vals))
        sig_max = float(np.max(valid_vals))
        # Saturation: values at or within 1% of the observed range boundary
        rng = sig_max - sig_min
        if rng > 0:
            sat_lo = sig_min + 0.01 * rng
            sat_hi = sig_max - 0.01 * rng
            sat_count = int(np.sum((valid_vals <= sat_lo) | (valid_vals >= sat_hi)))
            saturation_pct = 100.0 * sat_count / len(valid_vals)
        else:
            saturation_pct = 100.0
    else:
        mean_val = float("nan")
        std_val = float("nan")
        var_val = float("nan")
        sig_min = float("nan")
        sig_max = float("nan")
        saturation_pct = float("nan")

    # Timestamp jitter
    if len(timestamps) > 1:
        diffs = np.diff(timestamps)
        expected_dt = np.median(diffs)
        jitter = np.abs(diffs - expected_dt)
        ts_jitter_mean = float(np.mean(jitter))
        ts_jitter_max = float(np.max(jitter))
    else:
        ts_jitter_mean = 0.0
        ts_jitter_max = 0.0

    return {
        "n_samples": n,
        "missing_pct": round(missing_pct, 4),
        "interpolated_pct": round(interp_pct, 4),
        "outlier_pct": round(outlier_pct, 4),
        "nan_pct": round(nan_pct, 4),
        "mean": mean_val,
        "std": std_val,
        "variance": var_val,
        "signal_range": [sig_min, sig_max],
        "saturation_pct": round(saturation_pct, 4),
        "timestamp_jitter_mean_s": round(ts_jitter_mean, 8),
        "timestamp_jitter_max_s": round(ts_jitter_max, 8),
    }


def compute_all_quality(
    channel_values: Dict[str, np.ndarray],
    outlier_masks: Dict[str, np.ndarray],
    missing_masks: Dict[str, np.ndarray],
    interpolated_masks: Dict[str, np.ndarray],
    timestamps: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Compute quality metrics for all channels.

    Args:
        channel_values: Dict channel_name -> processed values.
        outlier_masks: Dict channel_name -> outlier boolean mask.
        missing_masks: Dict channel_name -> missing boolean mask.
        interpolated_masks: Dict channel_name -> interpolated boolean mask.
        timestamps: Shared timestamp grid.

    Returns:
        Dict channel_name -> quality metrics dict.
    """
    quality = {}
    n = len(timestamps)
    for ch_name, vals in channel_values.items():
        quality[ch_name] = compute_channel_quality(
            values=vals,
            outlier_mask=outlier_masks.get(ch_name, np.zeros(n, dtype=bool)),
            missing_mask=missing_masks.get(ch_name, np.zeros(n, dtype=bool)),
            interpolated_mask=interpolated_masks.get(ch_name, np.zeros(n, dtype=bool)),
            timestamps=timestamps,
        )
    return quality
