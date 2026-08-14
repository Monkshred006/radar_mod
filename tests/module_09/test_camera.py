"""Tests for VirtualCamera class."""

import numpy as np
import pytest

from module_09_3d.camera import VirtualCamera
from module_09_3d.config import CameraConfig


class TestCamera:
    def test_camera_matrices(self):
        cfg = CameraConfig(fov_degrees=45.0, image_width=256, image_height=256)
        cam = VirtualCamera(cfg)

        view = cam.get_view_matrix()
        proj = cam.get_projection_matrix()

        assert view.shape == (4, 4)
        assert proj.shape == (4, 4)

    def test_camera_position_and_target_updates(self):
        cam = VirtualCamera()
        cam.set_position([0.0, -10.0, 5.0])
        cam.set_target([1.0, 1.0, 1.0])

        assert np.allclose(cam.position, np.array([0.0, -10.0, 5.0]))
        assert np.allclose(cam.target, np.array([1.0, 1.0, 1.0]))

    def test_camera_project_method(self):
        cam = VirtualCamera(CameraConfig(camera_position=[0.0, -5.0, 0.0], camera_target=[0.0, 0.0, 0.0]))
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        coords, depths, valid = cam.project(pts)
        assert len(coords) == 2
        assert valid.all()
