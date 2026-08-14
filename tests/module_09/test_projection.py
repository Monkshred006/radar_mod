"""Tests for 3D to 2D projection mathematics."""

import numpy as np
import pytest

from module_09_3d.projection import (
    compute_look_at_matrix,
    compute_orthographic_matrix,
    compute_perspective_matrix,
    project_points,
)


class TestProjection:
    def test_look_at_matrix_shape_and_orthonormality(self):
        eye = np.array([0.0, -5.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])

        view = compute_look_at_matrix(eye, target, up)
        assert view.shape == (4, 4)
        # Rotation part (upper 3x3) should be orthogonal
        R = view[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)

    def test_perspective_matrix_shape(self):
        proj = compute_perspective_matrix(fov_degrees=60.0, aspect_ratio=1.0, near=0.1, far=100.0)
        assert proj.shape == (4, 4)
        assert proj[3, 2] == -1.0

    def test_project_points_in_front_of_camera(self):
        eye = np.array([0.0, -5.0, 0.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        view = compute_look_at_matrix(eye, target, up)
        proj = compute_perspective_matrix(45.0, 1.0, 0.1, 10.0)

        # Origin point is at distance 5 directly in front of camera
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        coords, depths, valid = project_points(pts, view, proj, 256, 256)

        assert valid[0]
        assert depths[0] == pytest.approx(5.0, abs=1e-2)
        # Projected near image center (127, 127)
        assert abs(coords[0, 0] - 127) <= 2
        assert abs(coords[0, 1] - 127) <= 2
