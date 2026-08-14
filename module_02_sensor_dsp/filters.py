"""Digital filtering utilities for sensor signal preprocessing.

Supports:
- Moving Average (causal for streaming, centred for offline)
- Exponential Moving Average (EMA) — always causal
- Median Filter (causal windowed)
- Low-pass IIR Butterworth filter (offline: filtfilt, streaming: lfilter)

All functions accept and return numpy arrays, preserving input dtype.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.signal import butter, filtfilt, lfilter, medfilt

from module_02_sensor_dsp.config import FilterConfig


def apply_moving_average(
    signal: np.ndarray,
    window: int = 5,
    causal: bool = True,
) -> np.ndarray:
    """Apply a moving average filter.

    Args:
        signal: 1-D array of signal values.
        window: Number of samples in the averaging window.
        causal: If True (streaming mode), use only past samples.
                If False (offline mode), use centred window.

    Returns:
        Smoothed signal of same length and dtype.
    """
    if window <= 1:
        return signal.copy()
    signal = signal.astype(np.float64)
    kernel = np.ones(window) / window
    if causal:
        # Pad left to keep length, avoid future samples
        padded = np.concatenate([np.full(window - 1, signal[0]), signal])
        return np.convolve(padded, kernel, mode="valid").astype(signal.dtype if signal.dtype != np.float64 else np.float64)
    else:
        return np.convolve(signal, kernel, mode="same")


def apply_ema(
    signal: np.ndarray,
    alpha: float = 0.1,
    initial_value: Optional[float] = None,
) -> np.ndarray:
    """Apply Exponential Moving Average (always causal).

    Args:
        signal: 1-D array.
        alpha: Smoothing factor in (0, 1]. Higher = faster response (less smoothing).
        initial_value: Starting EMA value. Defaults to signal[0].

    Returns:
        EMA-smoothed signal.
    """
    if len(signal) == 0:
        return signal.copy()
    out = np.empty_like(signal, dtype=np.float64)
    ema = float(signal[0]) if initial_value is None else initial_value
    for i, v in enumerate(signal):
        if np.isnan(v):
            out[i] = np.nan
        else:
            ema = alpha * float(v) + (1.0 - alpha) * ema
            out[i] = ema
    return out


def apply_median_filter(
    signal: np.ndarray,
    window: int = 5,
    causal: bool = True,
) -> np.ndarray:
    """Apply a median filter.

    Args:
        signal: 1-D array.
        window: Window size (must be odd; if even, incremented by 1).
        causal: If True, use only past samples (streaming safe).

    Returns:
        Median-filtered signal.
    """
    if window % 2 == 0:
        window += 1
    if not causal:
        return medfilt(signal.astype(np.float64), kernel_size=window)
    # Causal rolling median
    out = np.empty(len(signal), dtype=np.float64)
    half = window - 1  # how many past samples to include
    for i in range(len(signal)):
        start = max(0, i - half)
        out[i] = np.nanmedian(signal[start:i + 1])
    return out


def apply_lowpass(
    signal: np.ndarray,
    cutoff_hz: float,
    sampling_rate_hz: float,
    order: int = 2,
    causal: bool = True,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter.

    Args:
        signal: 1-D array.
        cutoff_hz: Cutoff frequency in Hz.
        sampling_rate_hz: Sampling rate in Hz.
        order: Filter order.
        causal: If True (streaming), use one-way lfilter (introduces phase lag).
                If False (offline), use zero-phase filtfilt.

    Returns:
        Filtered signal (float64).
    """
    nyquist = 0.5 * sampling_rate_hz
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)  # guard against >= 1
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    sig = signal.astype(np.float64)
    # Handle NaNs by forward-filling before filtering
    nan_mask = np.isnan(sig)
    if nan_mask.any():
        # Forward fill for filter stability
        sig_clean = sig.copy()
        last = sig_clean[0] if not nan_mask[0] else 0.0
        for i in range(len(sig_clean)):
            if nan_mask[i]:
                sig_clean[i] = last
            else:
                last = sig_clean[i]
    else:
        sig_clean = sig

    if causal:
        filtered = lfilter(b, a, sig_clean)
    else:
        filtered = filtfilt(b, a, sig_clean)

    # Restore NaN positions
    filtered[nan_mask] = np.nan
    return filtered


def apply_filter(
    signal: np.ndarray,
    config: FilterConfig,
    causal: bool = True,
) -> np.ndarray:
    """Dispatch to the correct filter based on FilterConfig.

    Args:
        signal: 1-D numpy array.
        config: FilterConfig specifying filter type and parameters.
        causal: Controls streaming vs offline mode.

    Returns:
        Filtered signal (float64).
    """
    ft = config.filter_type
    if ft == "none":
        return signal.astype(np.float64)
    elif ft == "moving_average":
        return apply_moving_average(signal, window=config.window, causal=causal)
    elif ft == "ema":
        return apply_ema(signal, alpha=config.alpha)
    elif ft == "median":
        return apply_median_filter(signal, window=config.window, causal=causal)
    elif ft == "lowpass":
        return apply_lowpass(
            signal,
            cutoff_hz=config.cutoff_hz,
            sampling_rate_hz=config.sampling_rate_hz,
            order=config.order,
            causal=causal,
        )
    else:
        raise ValueError(f"Unknown filter_type: {ft!r}")
