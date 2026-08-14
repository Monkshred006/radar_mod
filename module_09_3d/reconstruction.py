"""3D Reconstruction backends and demo generators for Module 9.

Provides:
- SyntheticGeometryReconstructor: Demo cube, sphere, vehicle point clouds (SYNTHETIC VISUALIZATION ONLY).
- PassThroughReconstructor: Creates Scene3D from raw_points or existing PointCloud.
- Factory function: build_reconstructor(config).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from module_09_3d.config import ReconstructionConfig
from module_09_3d.interfaces import SceneInput, ThreeDReconstructor
from module_09_3d.point_cloud import PointCloud
from module_09_3d.scene import Object3D, Scene3D


def generate_synthetic_cube(n_points: int = 500, size: float = 1.0) -> np.ndarray:
    """Generate points randomly distributed on the 6 faces of a cube."""
    rng = np.random.default_rng(42)
    pts = []
    per_face = n_points // 6
    half = size / 2.0

    for axis in range(3):
        for sign in [-half, half]:
            face_pts = rng.uniform(-half, half, size=(per_face, 3)).astype(np.float32)
            face_pts[:, axis] = sign
            pts.append(face_pts)

    return np.vstack(pts)


def generate_synthetic_sphere(n_points: int = 500, radius: float = 0.8) -> np.ndarray:
    """Generate points uniformly on the surface of a sphere."""
    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2 * np.pi, size=n_points).astype(np.float32)
    cos_theta = rng.uniform(-1, 1, size=n_points).astype(np.float32)
    theta = np.arccos(cos_theta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack([x, y, z])


def generate_synthetic_vehicle(n_points: int = 500) -> np.ndarray:
    """Generate a stylized vehicle-like point cloud (cabin + chassis + wheels)."""
    rng = np.random.default_rng(42)
    pts = []

    # Chassis: box [-1.5, 1.5] x [-0.8, 0.8] x [0.0, 0.5]
    n_chassis = int(n_points * 0.5)
    cx = rng.uniform(-1.5, 1.5, size=n_chassis)
    cy = rng.uniform(-0.8, 0.8, size=n_chassis)
    cz = rng.uniform(0.0, 0.5, size=n_chassis)
    pts.append(np.column_stack([cx, cy, cz]))

    # Cabin: box [-0.5, 0.8] x [-0.7, 0.7] x [0.5, 1.0]
    n_cabin = int(n_points * 0.3)
    k_x = rng.uniform(-0.5, 0.8, size=n_cabin)
    k_y = rng.uniform(-0.7, 0.7, size=n_cabin)
    k_z = rng.uniform(0.5, 1.0, size=n_cabin)
    pts.append(np.column_stack([k_x, k_y, k_z]))

    # Wheels (4 clusters)
    wheel_centers = [
        (-1.0, -0.85, 0.0), (-1.0, 0.85, 0.0),
        (1.0, -0.85, 0.0), (1.0, 0.85, 0.0)
    ]
    per_wheel = (n_points - n_chassis - n_cabin) // 4
    for wx, wy, wz in wheel_centers:
        w_pts = rng.normal(0, 0.15, size=(per_wheel, 3))
        w_pts[:, 0] += wx
        w_pts[:, 1] += wy
        w_pts[:, 2] += wz
        pts.append(w_pts)

    return np.vstack(pts).astype(np.float32)


class SyntheticGeometryReconstructor(ThreeDReconstructor):
    """Generates synthetic 3D geometry for visualization software validation.

    IMPORTANT: Labeled as SYNTHETIC VISUALIZATION DATA. Does not represent
    real radar 3D reconstruction accuracy.
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None) -> None:
        self.config = config or ReconstructionConfig()

    def reconstruct(self, scene_input: SceneInput) -> Scene3D:
        geom_type = self.config.synthetic_geometry_type
        n_pts = self.config.num_synthetic_points

        if geom_type == "cube":
            raw_pts = generate_synthetic_cube(n_pts)
            cls_id = 0
        elif geom_type == "sphere":
            raw_pts = generate_synthetic_sphere(n_pts)
            cls_id = 0
        elif geom_type == "vehicle":
            raw_pts = generate_synthetic_vehicle(n_pts)
            cls_id = 1
        elif geom_type == "multi_object":
            v1 = generate_synthetic_vehicle(n_pts // 2) + np.array([-1.5, 0, 0], dtype=np.float32)
            v2 = generate_synthetic_sphere(n_pts // 2, radius=0.5) + np.array([1.5, 0, 0], dtype=np.float32)
            raw_pts = np.vstack([v1, v2])
            cls_id = 1
        else:
            raw_pts = generate_synthetic_cube(n_pts)
            cls_id = 0

        # Optional noise
        if self.config.noise_std > 0:
            rng = np.random.default_rng(42)
            noise = rng.normal(0, self.config.noise_std, size=raw_pts.shape).astype(np.float32)
            raw_pts = raw_pts + noise

        # Assign confidences and classes
        confidences = np.ones(len(raw_pts), dtype=np.float32) * 0.95
        semantic_classes = np.full(len(raw_pts), cls_id, dtype=np.int32)

        pc = PointCloud(
            points=raw_pts,
            confidences=confidences,
            semantic_classes=semantic_classes,
            coordinate_system="right_handed_z_up",
            metadata={"source": "SYNTHETIC VISUALIZATION DATA", "geometry_type": geom_type},
        )

        objects = [
            Object3D(
                object_id=1,
                class_id=cls_id,
                position=pc.center(),
                confidence=0.95,
            )
        ]

        return Scene3D(
            point_cloud=pc,
            objects=objects,
            timestamp=scene_input.timestamp,
            coordinate_frame="sensor_local",
            metadata={
                "reconstruction_backend": "synthetic_geometry",
                "synthetic_notice": "SYNTHETIC VISUALIZATION DATA",
            },
        )


class PassThroughReconstructor(ThreeDReconstructor):
    """Builds a Scene3D directly from raw_points provided in SceneInput."""

    def reconstruct(self, scene_input: SceneInput) -> Scene3D:
        if scene_input.raw_points is None:
            # Fallback to single origin point
            pts = np.zeros((1, 3), dtype=np.float32)
        else:
            pts = np.asarray(scene_input.raw_points, dtype=np.float32)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.shape[1] > 3:
                pts = pts[:, :3]

        pc = PointCloud(
            points=pts,
            coordinate_system="right_handed_z_up",
            metadata={"source": "pass_through"},
        )
        return Scene3D(
            point_cloud=pc,
            timestamp=scene_input.timestamp,
            coordinate_frame="sensor_local",
            metadata={"reconstruction_backend": "passthrough"},
        )


def build_reconstructor(config: ReconstructionConfig) -> ThreeDReconstructor:
    """Factory: build the configured reconstructor backend."""
    if config.backend == "synthetic_geometry":
        return SyntheticGeometryReconstructor(config)
    elif config.backend == "passthrough":
        return PassThroughReconstructor()
    elif config.backend == "learned_radar_to_3d":
        raise NotImplementedError(
            "Learned radar-to-3D reconstructor backend is reserved for when real 3D "
            "ground truth and target representations are defined."
        )
    else:
        raise ValueError(f"Unknown reconstruction backend: '{config.backend}'")
