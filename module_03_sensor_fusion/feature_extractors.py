"""Generic mathematical/temporal feature extraction utilities.

All functions operate along time axis T while preserving temporal ordering.
Causal implementation ensures zero future leakage in streaming mode.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import numpy as np


def compute_first_difference(signal: np.ndarray, causal: bool = True) -> np.ndarray:
    """Compute first temporal difference: Δx_t = x_t - x_{t-1}.

    At t=0, Δx_0 = 0.
    """
    if len(signal) == 0:
        return np.zeros_like(signal)
    diff = np.zeros_like(signal, dtype=np.float64)
    diff[1:] = signal[1:] - signal[:-1]
    # Keep NaNs if original values were NaN
    nan_mask = np.isnan(signal)
    diff[nan_mask] = np.nan
    return diff


def compute_rate_of_change(
    signal: np.ndarray,
    timestamps: np.ndarray,
    causal: bool = True,
) -> np.ndarray:
    """Compute rate of change dx/dt = (x_t - x_{t-1}) / (t_t - t_{t-1}).

    At t=0, rate = 0.
    """
    if len(signal) <= 1:
        return np.zeros_like(signal, dtype=np.float64)
    dt = np.diff(timestamps)
    dx = np.diff(signal)
    rate = np.zeros_like(signal, dtype=np.float64)

    # Avoid division by zero dt
    valid_dt = dt > 0
    rate[1:][valid_dt] = dx[valid_dt] / dt[valid_dt]
    rate[np.isnan(signal)] = np.nan
    return rate


def compute_rolling_mean(
    signal: np.ndarray,
    window: int = 5,
    causal: bool = True,
) -> np.ndarray:
    """Compute causal rolling mean.

    At index t, averages signal[max(0, t-window+1) : t+1].
    """
    n = len(signal)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if causal:
            start = max(0, i - window + 1)
            seg = signal[start : i + 1]
        else:
            half = window // 2
            start = max(0, i - half)
            end = min(n, i + half + 1)
            seg = signal[start:end]
        valid = seg[~np.isnan(seg)]
        out[i] = np.mean(valid) if len(valid) > 0 else np.nan
    return out


def compute_rolling_std(
    signal: np.ndarray,
    window: int = 5,
    causal: bool = True,
) -> np.ndarray:
    """Compute causal rolling standard deviation."""
    n = len(signal)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if causal:
            start = max(0, i - window + 1)
            seg = signal[start : i + 1]
        else:
            half = window // 2
            start = max(0, i - half)
            end = min(n, i + half + 1)
            seg = signal[start:end]
        valid = seg[~np.isnan(seg)]
        out[i] = np.std(valid) if len(valid) > 1 else 0.0
    return out


def compute_rolling_energy(
    signal: np.ndarray,
    window: int = 5,
    causal: bool = True,
) -> np.ndarray:
    """Compute causal rolling energy: sum(x^2) over window."""
    n = len(signal)
    out = np.empty(n, dtype=np.float64)
    squared = signal ** 2
    for i in range(n):
        if causal:
            start = max(0, i - window + 1)
            seg = squared[start : i + 1]
        else:
            half = window // 2
            start = max(0, i - half)
            end = min(n, i + half + 1)
            seg = squared[start:end]
        valid = seg[~np.isnan(seg)]
        out[i] = np.sum(valid) if len(valid) > 0 else 0.0
    return out
