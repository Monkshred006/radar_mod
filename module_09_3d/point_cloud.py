"""PointCloud data structure for Module 9.

Supports:
- Nx3 spatial coordinates (x=lateral, y=depth, z=vertical)
- Optional per-point features: intensity, confidence, velocity, semantic_class
- Strict shape, finite value, and range validation
- Rigid transformations (rotation, translation, scaling) without destructive inplace corruption
- Bounding box computation, filtering, and normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class PointCloud:
    """Represents a 3D Point Cloud.

    Attributes
    ----------
    points : ndarray[N, 3]
        3D coordinates (float32).
    intensities : ndarray[N], optional
        Signal return intensity or reflectivity.
    confidences : ndarray[N], optional
        Confidence scores in range [0.0, 1.0].
    velocities : ndarray[N, 3] or ndarray[N], optional
        Doppler or radial/Cartesian velocity components.
    semantic_classes : ndarray[N], optional
        Integer class IDs per point.
    coordinate_system : str
        Metadata describing axis convention (default: 'right_handed_z_up').
    metadata : Dict[str, Any]
        Arbitrary extra metadata.
    """

    points: np.ndarray
    intensities: Optional[np.ndarray] = None
    confidences: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    semantic_classes: Optional[np.ndarray] = None
    coordinate_system: str = "right_handed_z_up"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points array must have shape (N, 3), got {self.points.shape}")

        if not np.isfinite(self.points).all():
            raise ValueError("PointCloud contains non-finite values (NaN or Inf).")

        n_pts = len(self.points)

        if self.intensities is not None:
            self.intensities = np.asarray(self.intensities, dtype=np.float32).reshape(-1)
            if len(self.intensities) != n_pts:
                raise ValueError(f"intensities length ({len(self.intensities)}) != points ({n_pts})")

        if self.confidences is not None:
            self.confidences = np.asarray(self.confidences, dtype=np.float32).reshape(-1)
            if len(self.confidences) != n_pts:
                raise ValueError(f"confidences length ({len(self.confidences)}) != points ({n_pts})")
            if (self.confidences < 0.0).any() or (self.confidences > 1.0).any():
                raise ValueError("confidences must be in range [0.0, 1.0]")

        if self.semantic_classes is not None:
            self.semantic_classes = np.asarray(self.semantic_classes, dtype=np.int32).reshape(-1)
            if len(self.semantic_classes) != n_pts:
                raise ValueError(f"semantic_classes length ({len(self.semantic_classes)}) != points ({n_pts})")

    @property
    def num_points(self) -> int:
        return len(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box (min_bound, max_bound)."""
        if self.is_empty():
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.points.min(axis=0), self.points.max(axis=0)

    def center(self) -> np.ndarray:
        """Compute geometric centroid of points."""
        if self.is_empty():
            return np.zeros(3, dtype=np.float32)
        return self.points.mean(axis=0)

    def transform(self, matrix_4x4: np.ndarray) -> "PointCloud":
        """Apply a 4x4 homogeneous transformation matrix to create a transformed copy."""
        if self.is_empty():
            return self.copy()

        pts_homo = np.hstack([self.points, np.ones((len(self.points), 1), dtype=np.float32)])
        transformed_homo = (matrix_4x4 @ pts_homo.T).T
        new_pts = transformed_homo[:, :3] / transformed_homo[:, 3:4]

        return PointCloud(
            points=new_pts.astype(np.float32),
            intensities=self.intensities.copy() if self.intensities is not None else None,
            confidences=self.confidences.copy() if self.confidences is not None else None,
            velocities=self.velocities.copy() if self.velocities is not None else None,
            semantic_classes=self.semantic_classes.copy() if self.semantic_classes is not None else None,
            coordinate_system=self.coordinate_system,
            metadata=dict(self.metadata),
        )

    def translate(self, offset: Union[List[float], np.ndarray]) -> "PointCloud":
        """Return translated copy."""
        offset = np.asarray(offset, dtype=np.float32).reshape(3)
        return PointCloud(
            points=self.points + offset,
            intensities=self.intensities.copy() if self.intensities is not None else None,
            confidences=self.confidences.copy() if self.confidences is not None else None,
            velocities=self.velocities.copy() if self.velocities is not None else None,
            semantic_classes=self.semantic_classes.copy() if self.semantic_classes is not None else None,
            coordinate_system=self.coordinate_system,
            metadata=dict(self.metadata),
        )

    def scale(self, factor: float) -> "PointCloud":
        """Return scaled copy."""
        return PointCloud(
            points=self.points * factor,
            intensities=self.intensities.copy() if self.intensities is not None else None,
            confidences=self.confidences.copy() if self.confidences is not None else None,
            velocities=self.velocities.copy() if self.velocities is not None else None,
            semantic_classes=self.semantic_classes.copy() if self.semantic_classes is not None else None,
            coordinate_system=self.coordinate_system,
            metadata=dict(self.metadata),
        )

    def filter_by_confidence(self, min_confidence: float) -> "PointCloud":
        """Filter points retaining only those with confidence >= min_confidence."""
        if self.confidences is None or self.is_empty():
            return self.copy()

        mask = self.confidences >= min_confidence
        return PointCloud(
            points=self.points[mask],
            intensities=self.intensities[mask] if self.intensities is not None else None,
            confidences=self.confidences[mask] if self.confidences is not None else None,
            velocities=self.velocities[mask] if self.velocities is not None else None,
            semantic_classes=self.semantic_classes[mask] if self.semantic_classes is not None else None,
            coordinate_system=self.coordinate_system,
            metadata=dict(self.metadata),
        )

    def copy(self) -> "PointCloud":
        return PointCloud(
            points=self.points.copy(),
            intensities=self.intensities.copy() if self.intensities is not None else None,
            confidences=self.confidences.copy() if self.confidences is not None else None,
            velocities=self.velocities.copy() if self.velocities is not None else None,
            semantic_classes=self.semantic_classes.copy() if self.semantic_classes is not None else None,
            coordinate_system=self.coordinate_system,
            metadata=dict(self.metadata),
        )
