"""Calibrated FMCW Radar Physics Mappings.

Provides explicit physical mapping functions between radar parameters:
- Radial velocity v (m/s) <-> Doppler frequency shift f_D (Hz)
- Range R (m) <-> Fast-time beat frequency f_b (Hz)
- Normalized bin coordinates <-> Physical metric units
"""

from __future__ import annotations

import torch
from module_06_physics.radar_constants import (
    C,
    FC,
    WAVELENGTH,
    CHIRP_SLOPE,
    MAX_RANGE,
    MIN_RANGE,
    MAX_VELOCITY,
    MIN_VELOCITY,
)


def velocity_to_doppler_shift(velocity: torch.Tensor | float) -> torch.Tensor | float:
    """Calculate Doppler frequency shift f_D from radial velocity v.

    Equation:
        f_D = 2 * v / lambda = 2 * v * f_c / c

    Args:
        velocity: Radial velocity in m/s.

    Returns:
        Doppler frequency shift in Hz.
    """
    return 2.0 * velocity / WAVELENGTH


def doppler_shift_to_velocity(doppler_shift: torch.Tensor | float) -> torch.Tensor | float:
    """Calculate radial velocity v from Doppler frequency shift f_D.

    Equation:
        v = f_D * lambda / 2

    Args:
        doppler_shift: Doppler frequency shift in Hz.

    Returns:
        Radial velocity in m/s.
    """
    return doppler_shift * WAVELENGTH / 2.0


def range_to_beat_frequency(target_range: torch.Tensor | float) -> torch.Tensor | float:
    """Calculate fast-time FMCW beat frequency f_b from target range R.

    Equation:
        f_b = 2 * S * R / c

    Args:
        target_range: Target range in meters.

    Returns:
        Beat frequency in Hz.
    """
    return 2.0 * CHIRP_SLOPE * target_range / C


def beat_frequency_to_range(beat_frequency: torch.Tensor | float) -> torch.Tensor | float:
    """Calculate target range R from fast-time beat frequency f_b.

    Equation:
        R = c * f_b / (2 * S)

    Args:
        beat_frequency: Beat frequency in Hz.

    Returns:
        Target range in meters.
    """
    return C * beat_frequency / (2.0 * CHIRP_SLOPE)


def normalized_to_physical_range(range_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalized range coordinate [0, 1] to physical meters [0.0, 15.0]."""
    return MIN_RANGE + range_norm * (MAX_RANGE - MIN_RANGE)


def normalized_to_physical_velocity(vel_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalized velocity coordinate [-1, 1] to physical m/s [-8.32, +8.32]."""
    return MIN_VELOCITY + (vel_norm + 1.0) * 0.5 * (MAX_VELOCITY - MIN_VELOCITY)
