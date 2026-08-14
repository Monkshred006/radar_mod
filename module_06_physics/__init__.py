"""PhotonShield AI Phase V2 Physics-Informed Module.

Provides calibrated radar physical constants, differentiable observable extractors,
FMCW mapping equations, and physics regularizers.
"""

from module_06_physics.radar_constants import (
    C,
    FC,
    WAVELENGTH,
    BANDWIDTH,
    CHIRP_DURATION,
    CHIRP_SLOPE,
    MAX_RANGE,
    MIN_RANGE,
    RANGE_RESOLUTION,
    MAX_VELOCITY,
    MIN_VELOCITY,
    VELOCITY_RESOLUTION,
    DT,
    FRAME_RATE_HZ,
)
from module_06_physics.observable_extractor import RadarObservableExtractor
from module_06_physics.fmcw_model import (
    velocity_to_doppler_shift,
    doppler_shift_to_velocity,
    range_to_beat_frequency,
    beat_frequency_to_range,
    normalized_to_physical_range,
    normalized_to_physical_velocity,
)
from module_06_physics.physics_losses import RadarPhysicsLoss
from module_06_physics.diagnostics import PhysicsDiagnostics

__all__ = [
    "C",
    "FC",
    "WAVELENGTH",
    "BANDWIDTH",
    "CHIRP_DURATION",
    "CHIRP_SLOPE",
    "MAX_RANGE",
    "MIN_RANGE",
    "RANGE_RESOLUTION",
    "MAX_VELOCITY",
    "MIN_VELOCITY",
    "VELOCITY_RESOLUTION",
    "DT",
    "FRAME_RATE_HZ",
    "RadarObservableExtractor",
    "velocity_to_doppler_shift",
    "doppler_shift_to_velocity",
    "range_to_beat_frequency",
    "beat_frequency_to_range",
    "normalized_to_physical_range",
    "normalized_to_physical_velocity",
    "RadarPhysicsLoss",
    "PhysicsDiagnostics",
]
