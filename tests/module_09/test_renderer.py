"""Tests for PointCloudRenderer."""

import numpy as np
import pytest

from module_09_3d.camera import VirtualCamera
from module_09_3d.config import CameraConfig, RenderConfig
from module_09_3d.point_cloud import PointCloud
from module_09_3d.renderer import PointCloudRenderer
from module_09_3d.scene import Scene3D


class TestRenderer:
    def test_render_output_shape_and_dtype(self):
        cam = VirtualCamera(CameraConfig(image_width=128, image_height=128))
        renderer = PointCloudRenderer(cam, RenderConfig(color_mode="depth"))

        pts = np.random.uniform(-1, 1, size=(50, 3)).astype(np.float32)
        pc = PointCloud(points=pts)

        img = renderer.render(pc)
        assert img.shape == (128, 128, 3)
        assert img.dtype == np.uint8

    def test_empty_point_cloud_renders_background(self):
        renderer = PointCloudRenderer(
            config=RenderConfig(background_color=(0, 0, 0))
        )
        empty_pc = PointCloud(points=np.zeros((0, 3), dtype=np.float32))

        img = renderer.render(empty_pc)
        assert (img == 0).all()

    @pytest.mark.parametrize("color_mode", ["solid", "depth", "semantic_class", "confidence"])
    def test_renderer_color_modes(self, color_mode):
        cfg = RenderConfig(color_mode=color_mode)
        renderer = PointCloudRenderer(config=cfg)

        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)
        confs = np.array([0.9, 0.4], dtype=np.float32)
        classes = np.array([1, 2], dtype=np.int32)
        pc = PointCloud(points=pts, confidences=confs, semantic_classes=classes)

        img = renderer.render(pc)
        assert img.shape == (256, 256, 3)

    def test_render_scene_object(self):
        renderer = PointCloudRenderer()
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        scene = Scene3D(point_cloud=PointCloud(points=pts))

        img = renderer.render(scene)
        assert img.shape == (256, 256, 3)
