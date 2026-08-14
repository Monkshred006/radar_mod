"""Validation logic for radar data integrity and sequence checking."""

from pathlib import Path
from typing import Union, Sequence, Optional
import numpy as np


class RadarDataValidationError(Exception):
    """Base exception for radar data validation failures."""
    pass


class CorruptedFileError(RadarDataValidationError):
    """Raised when a radar data file cannot be parsed or is corrupted."""
    pass


class InconsistentDimensionError(RadarDataValidationError):
    """Raised when frame or sequence dimensions do not match expected shapes."""
    pass


def validate_frame(frame: np.ndarray, expected_shape: Optional[tuple] = None) -> None:
    """Validate a single raw radar frame numpy array.

    Checks:
    - Array is non-empty
    - No NaN values
    - No Infinite values
    - Expected dimensions (if expected_shape specified)

    Raises:
        RadarDataValidationError or InconsistentDimensionError on validation failure.
    """
    if not isinstance(frame, np.ndarray):
        raise RadarDataValidationError(f"Expected numpy.ndarray frame, got {type(frame)}")

    if frame.size == 0:
        raise RadarDataValidationError("Radar frame array is empty (0 size).")

    # Check NaN and Inf values
    if np.isnan(frame).any():
        raise RadarDataValidationError("Radar frame contains NaN (Not-a-Number) values.")

    if np.isinf(frame).any():
        raise RadarDataValidationError("Radar frame contains Infinite (Inf) values.")

    if expected_shape is not None and frame.shape != expected_shape:
        raise InconsistentDimensionError(
            f"Frame shape mismatch: expected {expected_shape}, got {frame.shape}"
        )


def validate_sequence(sequence: Sequence[np.ndarray]) -> tuple:
    """Validate a sequence of raw radar frame arrays.

    Checks:
    - Sequence is non-empty
    - All frames in sequence have identical dimensions
    - All individual frames pass validate_frame checks

    Returns:
        tuple: Unified frame dimension shape (excluding sequence length).
    """
    if len(sequence) == 0:
        raise RadarDataValidationError("Empty radar frame sequence provided.")

    first_shape = None
    for idx, frame in enumerate(sequence):
        validate_frame(frame)
        if first_shape is None:
            first_shape = frame.shape
        elif frame.shape != first_shape:
            raise InconsistentDimensionError(
                f"Inconsistent frame dimension within sequence at index {idx}: "
                f"expected {first_shape}, got {frame.shape}"
            )

    return first_shape


def validate_file_exists(filepath: Union[str, Path]) -> Path:
    """Ensure specified path exists and is a valid file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Radar data file does not exist: {path}")
    if not path.is_file():
        raise RadarDataValidationError(f"Path is not a regular file: {path}")
    return path
