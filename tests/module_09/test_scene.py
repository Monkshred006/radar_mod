"""Tests for Scene3D and Object3D representations."""

import numpy as np
import pytest

from module_09_3d.point_cloud import PointCloud
from module_09_3d.scene import Object3D, Scene3D


class TestScene:
    def test_scene_creation(self):
        pc = PointCloud(points=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
        obj = Object3D(object_id=1, class_id=2, position=np.array([0.5, 0.5, 0.5]))
        scene = Scene3D(point_cloud=pc, objects=[obj], timestamp=10.0)

        assert scene.point_cloud.num_points == 2
        assert len(scene.objects) == 1
        assert scene.timestamp == 10.0

    def test_scene_transform(self):
        pc = PointCloud(points=np.array([[1, 0, 0]], dtype=np.float32))
        obj = Object3D(object_id=1, class_id=0, position=np.array([1, 0, 0]))
        scene = Scene3D(point_cloud=pc, objects=[obj])

        # Translation matrix: +5 on Z axis
        T = np.eye(4, dtype=np.float32)
        T[2, 3] = 5.0

        trans_scene = scene.transform(T)
        assert np.allclose(trans_scene.point_cloud.points, np.array([[1, 0, 5]]))
        assert np.allclose(trans_scene.objects[0].position, np.array([1, 0, 5]))

    def test_scene_to_dict_serialization(self):
        pc = PointCloud(points=np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32))
        obj = Object3D(object_id=42, class_id=1, position=np.array([1, 1, 1]))
        scene = Scene3D(point_cloud=pc, objects=[obj], timestamp=5.5)

        data = scene.to_dict()
        assert data["num_points"] == 2
        assert data["timestamp"] == 5.5
        assert data["bounding_box_max"] == [2.0, 2.0, 2.0]
        assert data["objects"][0]["id"] == 42
