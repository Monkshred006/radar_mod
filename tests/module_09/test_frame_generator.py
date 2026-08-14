"""Tests for RotatingViewGenerator."""

import numpy as np
import pytest

from module_09_3d.config import CameraConfig, RenderConfig, RotationConfig
from module_09_3d.frame_generator import RotatingViewGenerator
from module_09_3d.point_cloud import PointCloud
from module_09_3d.scene import Scene3D


class TestFrameGenerator:
    def test_frame_count_and_shape(self):
        rot_cfg = RotationConfig(rotation_step_deg=45.0)  # 360/45 = 8 frames
        cam_cfg = CameraConfig(image_width=64, image_height=64)
        generator = RotatingViewGenerator(rotation_config=rot_cfg, camera_config=cam_cfg)

        pts = np.random.uniform(-1, 1, size=(20, 3)).astype(np.float32)
        scene = Scene3D(point_cloud=PointCloud(points=pts))

        frames = generator.generate_frames(scene)
        assert len(frames) == 8
        assert frames[0].shape == (64, 64, 3)

    def test_non_destructive_scene_points(self):
        generator = RotatingViewGenerator()
        pts_orig = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scene = Scene3D(point_cloud=PointCloud(points=pts_orig.copy()))

        _ = generator.generate_frames(scene)
        # Verify points are bit-for-bit unchanged
        assert np.array_equal(scene.point_cloud.points, pts_orig)

    def test_generate_frames_with_metadata(self):
        rot_cfg = RotationConfig(rotation_step_deg=90.0)
        generator = RotatingViewGenerator(rotation_config=rot_cfg)

        pts = np.zeros((5, 3), dtype=np.float32)
        pc = PointCloud(points=pts)

        pairs = generator.generate_frames_with_metadata(pc)
        assert len(pairs) == 4
        assert [p[0] for p in pairs] == [0.0, 90.0, 180.0, 270.0]
