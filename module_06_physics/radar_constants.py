"""Radar physical and sensor constants for PhotonShield AI Phase V2.

Source metadata: data/radical/metadata/dataset_spec.json
Sensor: Texas Instruments IWR1443 mmWave 77 GHz Radar
"""

from __future__ import annotations

# Fundamental Physical Constants
C: float = 299792458.0  # Speed of light in vacuum (m/s)

# Sensor Hardware Parameters (TI IWR1443)
FC: float = 77.0e9  # Carrier frequency: 77.0 GHz
WAVELENGTH: float = C / FC  # Carrier wavelength: ~3.8934 mm (0.0038934 m)

BANDWIDTH: float = 4000.0e6  # Sweep bandwidth: 4.0 GHz
CHIRP_DURATION: float = 60.0e-6  # Chirp duration: 60.0 microseconds
CHIRP_SLOPE: float = BANDWIDTH / CHIRP_DURATION  # Chirp slope S: 6.667e13 Hz/s

# Range Specification
MAX_RANGE: float = 15.0  # Maximum unambiguous range: 15.0 m
MIN_RANGE: float = 0.0  # Minimum range: 0.0 m
RANGE_RESOLUTION: float = 0.15  # Range resolution: 0.15 m
NUM_RANGE_BINS: int = 64  # Raw Range-FFT bins in Range-Doppler map
NUM_RANGE_FEATS: int = 30  # Extracted Range profile feature slice: indices [0:30]

# Doppler / Velocity Specification
MAX_VELOCITY: float = 8.32  # Maximum unambiguous velocity: +8.32 m/s
MIN_VELOCITY: float = -8.32  # Minimum unambiguous velocity: -8.32 m/s
VELOCITY_RESOLUTION: float = 0.26  # Doppler velocity resolution: 0.26 m/s
NUM_DOPPLER_BINS: int = 32  # Raw Doppler-FFT bins in Range-Doppler map
NUM_DOPPLER_FEATS: int = 30  # Extracted Doppler profile feature slice: indices [30:60]

# Summary Scalar Indices
SUMMARY_SCALAR_FEATS: int = 4  # Indices [60:64] -> max_val, mean_val, std_val, snr_est

# Temporal Sampling
FRAME_RATE_HZ: float = 30.0  # RaDICaL sensor recording frame rate (30 FPS)
DT: float = 1.0 / FRAME_RATE_HZ  # Inter-frame sampling interval: ~0.03333 seconds (33.33 ms)
