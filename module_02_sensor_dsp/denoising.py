"""Denoising and baseline correction utilities.

Primarily designed for BPW34 photodiode channels but sensor-agnostic.

Baseline estimation methods:
- EMA tracking: slow exponential moving average tracks DC offset / drift.
- Percentile rolling: rolling low-percentile as baseline estimate.

Noise suppression is delegated to filters.py.
"""

from __future__ import annotations
import numpy as np
from module_02_sensor_dsp.config import BaselineConfig


def estimate_baseline_ema(
    signal: np.ndarray,
    alpha: float = 0.005,
) -> np.ndarray:
    """Estimate signal baseline using a slow causal EMA.

    A very small alpha (e.g. 0.005) produces a slowly varying baseline
    that tracks DC drift and low-frequency offset without following the
    fast signal dynamics.

    Args:
        signal: 1-D float array.
        alpha: EMA smoothing factor (small = slow tracking).

    Returns:
        Baseline estimate array of the same shape.
    """
    baseline = np.empty_like(signal, dtype=np.float64)
    ema = float(np.nanmean(signal[:max(1, len(signal) // 10)]))  # warm start
    for i, v in enumerate(signal):
        if not np.isnan(v):
            ema = alpha * float(v) + (1.0 - alpha) * ema
        baseline[i] = ema
    return baseline


def estimate_baseline_percentile(
    signal: np.ndarray,
    window: int = 50,
    percentile: float = 5.0,
    causal: bool = True,
) -> np.ndarray:
    """Estimate baseline using a rolling low-percentile.

    Args:
        signal: 1-D float array.
        window: Rolling window size.
        percentile: Percentile value to use as baseline (e.g. 5 = 5th percentile).
        causal: If True, use only past samples (no lookahead).

    Returns:
        Baseline estimate array.
    """
    n = len(signal)
    baseline = np.empty(n, dtype=np.float64)
    for i in range(n):
        if causal:
            start = max(0, i - window + 1)
            segment = signal[start:i + 1]
        else:
            half = window // 2
            start = max(0, i - half)
            end = min(n, i + half + 1)
            segment = signal[start:end]
        valid = segment[~np.isnan(segment)]
        baseline[i] = np.percentile(valid, percentile) if len(valid) > 0 else 0.0
    return baseline


def remove_baseline(
    signal: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Subtract baseline from signal.

    Args:
        signal: 1-D float array.
        baseline: 1-D baseline estimate (same length).

    Returns:
        Baseline-corrected signal.
    """
    return signal.astype(np.float64) - baseline


def apply_baseline_correction(
    signal: np.ndarray,
    config: BaselineConfig,
    causal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply baseline estimation and removal.

    Args:
        signal: 1-D float array.
        config: BaselineConfig settings.
        causal: Enforce causal processing (streaming safe).

    Returns:
        Tuple of (corrected_signal, baseline_estimate).
        If config.enabled is False, returns (signal copy, zeros array).
    """
    if not config.enabled:
        return signal.astype(np.float64), np.zeros_like(signal, dtype=np.float64)

    if config.method == "ema":
        baseline = estimate_baseline_ema(signal, alpha=config.alpha)
    elif config.method == "percentile":
        baseline = estimate_baseline_percentile(
            signal,
            window=config.window,
            percentile=config.percentile,
            causal=causal,
        )
    else:
        raise ValueError(f"Unknown baseline method: {config.method!r}")

    corrected = remove_baseline(signal, baseline)
    return corrected, baseline
