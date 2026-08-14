"""Scene3D and Object3D representations for Module 9."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from module_09_3d.point_cloud import PointCloud


@dataclass
class Object3D:
    """Optional object-level detection/tracking entity in 3D space.

    Attributes
    ----------
    object_id : int
    class_id : int
    position : ndarray[3]
        Centroid or anchor location.
    velocity : ndarray[3], optional
        Estimated 3D velocity vector.
    confidence : float
    bounding_box : Tuple[ndarray[3], ndarray[3]], optional
        (min_corner, max_corner).
    metadata : Dict[str, Any]
    """

    object_id: int
    class_id: int
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    confidence: float = 1.0
    bounding_box: Optional[Tuple[np.ndarray, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float32).reshape(3)


@dataclass
class Scene3D:
    """Represents a 3D scene state at a single timestep.

    Attributes
    ----------
    point_cloud : PointCloud
        Primary geometric representation.
    objects : List[Object3D]
        Optional detected/tracked object entities.
    timestamp : float
    coordinate_frame : str
        e.g. 'sensor_local', 'world'.
    metadata : Dict[str, Any]
    """

    point_cloud: PointCloud
    objects: List[Object3D] = field(default_factory=list)
    timestamp: float = 0.0
    coordinate_frame: str = "sensor_local"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the total bounding box of the scene point cloud."""
        return self.point_cloud.bounding_box()

    def transform(self, matrix_4x4: np.ndarray) -> "Scene3D":
        """Transform all points and object positions by a 4x4 matrix."""
        transformed_pc = self.point_cloud.transform(matrix_4x4)
        transformed_objs = []
        for obj in self.objects:
            pos_h = np.array([obj.position[0], obj.position[1], obj.position[2], 1.0], dtype=np.float32)
            new_pos = (matrix_4x4 @ pos_h)[:3]
            transformed_objs.append(
                Object3D(
                    object_id=obj.object_id,
                    class_id=obj.class_id,
                    position=new_pos,
                    velocity=obj.velocity,
                    confidence=obj.confidence,
                    metadata=dict(obj.metadata),
                )
            )
        return Scene3D(
            point_cloud=transformed_pc,
            objects=transformed_objs,
            timestamp=self.timestamp,
            coordinate_frame=self.coordinate_frame,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata and summary statistics for export/logging."""
        min_b, max_b = self.bounding_box()
        return {
            "timestamp": self.timestamp,
            "coordinate_frame": self.coordinate_frame,
            "num_points": self.point_cloud.num_points,
            "bounding_box_min": min_b.tolist(),
            "bounding_box_max": max_b.tolist(),
            "num_objects": len(self.objects),
            "objects": [
                {
                    "id": obj.object_id,
                    "class": obj.class_id,
                    "pos": obj.position.tolist(),
                    "confidence": obj.confidence,
                }
                for obj in self.objects
            ],
            "metadata": self.metadata,
        }
