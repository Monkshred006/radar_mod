"""Tests for 3D reconstruction backends."""

import numpy as np
import pytest

from module_09_3d.config import ReconstructionConfig
from module_09_3d.interfaces import SceneInput
from module_09_3d.reconstruction import (
    PassThroughReconstructor,
    SyntheticGeometryReconstructor,
    build_reconstructor,
)


class TestReconstruction:
    @pytest.mark.parametrize("geom_type", ["cube", "sphere", "vehicle", "multi_object"])
    def test_synthetic_geometry_reconstructor(self, geom_type):
        cfg = ReconstructionConfig(backend="synthetic_geometry", synthetic_geometry_type=geom_type, num_synthetic_points=200)
        reconstructor = SyntheticGeometryReconstructor(cfg)

        scene_inp = SceneInput(timestamp=1.0)
        scene = reconstructor.reconstruct(scene_inp)

        assert scene.point_cloud.num_points >= 100
        assert np.isfinite(scene.point_cloud.points).all()
        assert scene.metadata["synthetic_notice"] == "SYNTHETIC VISUALIZATION DATA"

    def test_passthrough_reconstructor(self):
        reconstructor = PassThroughReconstructor()
        raw_pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        scene_inp = SceneInput(raw_points=raw_pts, timestamp=2.0)

        scene = reconstructor.reconstruct(scene_inp)
        assert scene.point_cloud.num_points == 2
        assert np.allclose(scene.point_cloud.points, raw_pts)

    def test_factory_learned_raises_not_implemented(self):
        cfg = ReconstructionConfig(backend="learned_radar_to_3d")
        with pytest.raises(NotImplementedError):
            build_reconstructor(cfg)
