"""Outlier detection utilities.

Outliers are FLAGGED, never silently removed.

Each function returns a boolean mask (True = outlier) alongside the original values.
Downstream components decide how to handle flagged values.

Methods supported:
- IQR (Interquartile Range)
- Z-Score
- MAD (Median Absolute Deviation)
- Range (physical validity bounds)
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from module_02_sensor_dsp.config import OutlierConfig


def detect_outliers_iqr(
    signal: np.ndarray,
    threshold: float = 1.5,
) -> np.ndarray:
    """Flag outliers using IQR method.

    A sample is flagged if it lies outside [Q1 - threshold*IQR, Q3 + threshold*IQR].

    Args:
        signal: 1-D float array (NaN values are not flagged as outliers).
        threshold: IQR multiplier (default 1.5; use 3.0 for extreme outliers).

    Returns:
        Boolean array (True = outlier, False = valid).
    """
    valid = signal[~np.isnan(signal)]
    if len(valid) < 4:
        return np.zeros(len(signal), dtype=bool)
    q1 = np.percentile(valid, 25)
    q3 = np.percentile(valid, 75)
    iqr = q3 - q1
    lo = q1 - threshold * iqr
    hi = q3 + threshold * iqr
    mask = np.zeros(len(signal), dtype=bool)
    valid_idx = ~np.isnan(signal)
    mask[valid_idx] = (signal[valid_idx] < lo) | (signal[valid_idx] > hi)
    return mask


def detect_outliers_zscore(
    signal: np.ndarray,
    threshold: float = 3.0,
) -> np.ndarray:
    """Flag outliers using Z-score.

    Args:
        signal: 1-D float array.
        threshold: Number of standard deviations.

    Returns:
        Boolean outlier mask.
    """
    valid = signal[~np.isnan(signal)]
    if len(valid) < 2:
        return np.zeros(len(signal), dtype=bool)
    mean = np.mean(valid)
    std = np.std(valid)
    if std == 0:
        return np.zeros(len(signal), dtype=bool)
    mask = np.zeros(len(signal), dtype=bool)
    valid_idx = ~np.isnan(signal)
    z = np.abs((signal[valid_idx] - mean) / std)
    mask[valid_idx] = z > threshold
    return mask


def detect_outliers_mad(
    signal: np.ndarray,
    threshold: float = 3.5,
) -> np.ndarray:
    """Flag outliers using Median Absolute Deviation (MAD).

    Uses the modified z-score:  0.6745 * |x - median| / MAD > threshold.

    Args:
        signal: 1-D float array.
        threshold: Modified z-score threshold (Iglewicz & Hoaglin recommend 3.5).

    Returns:
        Boolean outlier mask.
    """
    valid = signal[~np.isnan(signal)]
    if len(valid) < 3:
        return np.zeros(len(signal), dtype=bool)
    median = np.median(valid)
    mad = np.median(np.abs(valid - median))
    if mad == 0:
        # Fall back to z-score when MAD is zero
        return detect_outliers_zscore(signal, threshold=threshold)
    mask = np.zeros(len(signal), dtype=bool)
    valid_idx = ~np.isnan(signal)
    modified_z = 0.6745 * np.abs(signal[valid_idx] - median) / mad
    mask[valid_idx] = modified_z > threshold
    return mask


def detect_outliers_range(
    signal: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """Flag samples outside physical validity bounds.

    Args:
        signal: 1-D float array.
        min_val: Minimum valid value (None = no lower bound).
        max_val: Maximum valid value (None = no upper bound).

    Returns:
        Boolean outlier mask.
    """
    mask = np.zeros(len(signal), dtype=bool)
    valid_idx = ~np.isnan(signal)
    if min_val is not None:
        mask[valid_idx] |= signal[valid_idx] < min_val
    if max_val is not None:
        mask[valid_idx] |= signal[valid_idx] > max_val
    return mask


def detect_outliers(
    signal: np.ndarray,
    config: OutlierConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch outlier detection based on OutlierConfig.

    Args:
        signal: 1-D float array.
        config: OutlierConfig with method and parameters.

    Returns:
        Tuple of (signal_copy, outlier_mask).
        outlier_mask: boolean array (True = outlier).
    """
    method = config.method
    if method == "none":
        return signal.astype(np.float64), np.zeros(len(signal), dtype=bool)
    elif method == "iqr":
        mask = detect_outliers_iqr(signal, threshold=config.threshold)
    elif method == "zscore":
        mask = detect_outliers_zscore(signal, threshold=config.threshold)
    elif method == "mad":
        mask = detect_outliers_mad(signal, threshold=config.threshold)
    elif method == "range":
        mask = detect_outliers_range(
            signal, min_val=config.min_val, max_val=config.max_val
        )
    else:
        raise ValueError(f"Unknown outlier method: {method!r}")

    return signal.astype(np.float64), mask
