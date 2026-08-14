"""Rotating view generator for 360-degree scene visualization."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from module_09_3d.camera import VirtualCamera
from module_09_3d.config import CameraConfig, RenderConfig, RotationConfig
from module_09_3d.point_cloud import PointCloud
from module_09_3d.renderer import PointCloudRenderer
from module_09_3d.rotation import RotationTrajectory
from module_09_3d.scene import Scene3D


class RotatingViewGenerator:
    """Generates a complete 360-degree sequence of 2D rendered views for a Scene3D."""

    def __init__(
        self,
        rotation_config: Optional[RotationConfig] = None,
        camera_config: Optional[CameraConfig] = None,
        render_config: Optional[RenderConfig] = None,
    ) -> None:
        self.rotation_config = rotation_config or RotationConfig()
        self.camera_config = camera_config or CameraConfig()
        self.render_config = render_config or RenderConfig()

        self.trajectory = RotationTrajectory(self.rotation_config)
        self.camera = VirtualCamera(self.camera_config)
        self.renderer = PointCloudRenderer(self.camera, self.render_config)

    def generate_frames(
        self,
        scene: Scene3D | PointCloud,
    ) -> List[np.ndarray]:
        """Generate ordered list of RGB image frames for 360 rotation without mutating scene points.

        Returns
        -------
        frames : List[ndarray[H, W, 3], uint8]
        """
        # Determine center of scene to orbit around
        if isinstance(scene, Scene3D):
            target = scene.point_cloud.center()
        else:
            target = scene.center()

        self.camera.set_target(target)
        positions = self.trajectory.get_camera_positions(target)

        frames: List[np.ndarray] = []
        for azimuth_deg, cam_pos in positions:
            self.camera.set_position(cam_pos)
            frame = self.renderer.render(scene)
            frames.append(frame)

        return frames

    def generate_frames_with_metadata(
        self,
        scene: Scene3D | PointCloud,
    ) -> List[Tuple[float, np.ndarray]]:
        """Generate (azimuth_deg, frame) pairs."""
        if isinstance(scene, Scene3D):
            target = scene.point_cloud.center()
        else:
            target = scene.center()

        self.camera.set_target(target)
        positions = self.trajectory.get_camera_positions(target)

        output: List[Tuple[float, np.ndarray]] = []
        for azimuth_deg, cam_pos in positions:
            self.camera.set_position(cam_pos)
            frame = self.renderer.render(scene)
            output.append((azimuth_deg, frame))

        return output
