"""Tests for TrapezoidalPrismRenderer."""

import numpy as np
import pytest

from module_09_3d.config import PrismConfig
from module_09_3d.point_cloud import PointCloud
from module_09_3d.prism_display import TrapezoidalPrismRenderer
from module_09_3d.scene import Scene3D


class TestPrismDisplay:
    def test_render_prism_canvas_shape_and_type(self):
        cfg = PrismConfig(canvas_width=256, canvas_height=256, view_scale=0.3)
        prism_renderer = TrapezoidalPrismRenderer(prism_config=cfg)

        pts = np.random.uniform(-0.5, 0.5, size=(100, 3)).astype(np.float32)
        scene = Scene3D(point_cloud=PointCloud(points=pts))

        canvas = prism_renderer.render_prism_canvas(scene)
        assert canvas.shape == (256, 256, 3)
        assert canvas.dtype == np.uint8
        # Should have rendered non-black pixels from the point cloud
        assert (canvas > 0).any()

    def test_prism_renderer_with_point_cloud_direct(self):
        prism_renderer = TrapezoidalPrismRenderer()
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        pc = PointCloud(points=pts)

        canvas = prism_renderer.render_prism_canvas(pc)
        assert canvas.shape == (512, 512, 3)
