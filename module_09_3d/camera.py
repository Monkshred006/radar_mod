"""Virtual camera representation and matrix generation for Module 9."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from module_09_3d.config import CameraConfig
from module_09_3d.projection import (
    compute_look_at_matrix,
    compute_orthographic_matrix,
    compute_perspective_matrix,
    project_points,
)


class VirtualCamera:
    """Configurable 3D Virtual Camera for rendering scenes."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or CameraConfig()
        self.position = np.array(self.config.camera_position, dtype=np.float32)
        self.target = np.array(self.config.camera_target, dtype=np.float32)
        self.up = np.array(self.config.camera_up, dtype=np.float32)

    def set_position(self, pos: Union[List[float], np.ndarray]) -> None:
        self.position = np.asarray(pos, dtype=np.float32).reshape(3)

    def set_target(self, target: Union[List[float], np.ndarray]) -> None:
        self.target = np.asarray(target, dtype=np.float32).reshape(3)

    def get_view_matrix(self) -> np.ndarray:
        return compute_look_at_matrix(self.position, self.target, self.up)

    def get_projection_matrix(self) -> np.ndarray:
        aspect = self.config.image_width / max(self.config.image_height, 1)
        if self.config.projection_type == "perspective":
            return compute_perspective_matrix(
                fov_degrees=self.config.fov_degrees,
                aspect_ratio=aspect,
                near=self.config.near_clip,
                far=self.config.far_clip,
            )
        else:
            half_w = 2.0 * aspect
            half_h = 2.0
            return compute_orthographic_matrix(
                left=-half_w, right=half_w,
                bottom=-half_h, top=half_h,
                near=self.config.near_clip, far=self.config.far_clip,
            )

    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points to 2D pixel coordinates and depths."""
        view = self.get_view_matrix()
        proj = self.get_projection_matrix()
        return project_points(
            points_3d, view, proj,
            self.config.image_width,
            self.config.image_height,
        )
