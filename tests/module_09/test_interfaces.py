"""Tests for Module 9 interfaces and SceneInput data structure."""

import numpy as np
import pytest
import torch

from module_09_3d.interfaces import (
    DisplayBackend,
    ReconstructionEvaluator,
    SceneInput,
    ThreeDReconstructor,
)
from module_09_3d.point_cloud import PointCloud
from module_09_3d.scene import Scene3D


class DummyEvaluator(ReconstructionEvaluator):
    def evaluate(self, predicted_scene, ground_truth_scene=None):
        if ground_truth_scene is None:
            return {"status": "ground_truth_unavailable"}
        return {"status": "evaluated", "chamfer_distance": 0.0}


class TestInterfaces:
    def test_scene_input_defaults(self):
        inp = SceneInput()
        assert inp.latent is None
        assert inp.target_probability is None
        assert inp.physical_state is None
        assert inp.raw_points is None
        assert inp.timestamp == 0.0

    def test_scene_input_full_payload(self):
        latent = torch.randn(128)
        phys_state = np.array([1.0, 2.0, 0.0])
        raw_pts = np.random.randn(10, 3)

        inp = SceneInput(
            latent=latent,
            target_probability=0.9,
            anomaly_probability=0.05,
            environmental_assessment=[20.0, 50.0, 1013.0],
            physical_state=phys_state,
            raw_points=raw_pts,
            timestamp=123.45,
            frame_id=42,
        )

        assert inp.target_probability == 0.9
        assert inp.frame_id == 42
        assert len(inp.raw_points) == 10

    def test_reconstruction_evaluator_ground_truth_unavailable(self):
        evaluator = DummyEvaluator()
        pc = PointCloud(points=np.zeros((4, 3)))
        scene = Scene3D(point_cloud=pc)

        res = evaluator.evaluate(scene, ground_truth_scene=None)
        assert res["status"] == "ground_truth_unavailable"
