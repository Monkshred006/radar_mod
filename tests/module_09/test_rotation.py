"""Tests for RotationTrajectory."""

import numpy as np
import pytest

from module_09_3d.config import RotationConfig
from module_09_3d.rotation import RotationTrajectory


class TestRotation:
    def test_unique_angles_360_span(self):
        # Requirement #25: start=0, end=360, step=30 -> 12 unique angles without 360 duplicate
        cfg = RotationConfig(rotation_start_deg=0.0, rotation_end_deg=360.0, rotation_step_deg=30.0)
        traj = RotationTrajectory(cfg)

        angles = traj.get_angles_degrees()
        assert len(angles) == 12
        assert angles[0] == 0.0
        assert angles[-1] == 330.0
        assert 360.0 not in angles

    def test_partial_rotation_span(self):
        cfg = RotationConfig(rotation_start_deg=0.0, rotation_end_deg=90.0, rotation_step_deg=15.0)
        traj = RotationTrajectory(cfg)

        angles = traj.get_angles_degrees()
        assert len(angles) == 6  # 0, 15, 30, 45, 60, 75

    def test_camera_positions_distance_and_elevation(self):
        cfg = RotationConfig(distance=4.0, elevation_deg=30.0, rotation_step_deg=90.0)
        traj = RotationTrajectory(cfg)
        positions = traj.get_camera_positions(target=np.array([0, 0, 0], dtype=np.float32))

        assert len(positions) == 4
        for deg, pos in positions:
            # Distance from origin should be exactly 4.0
            dist = np.linalg.norm(pos)
            assert dist == pytest.approx(4.0, rel=1e-3)
            # Z should be 4.0 * sin(30°) = 2.0
            assert pos[2] == pytest.approx(2.0, rel=1e-3)
