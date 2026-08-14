"""Tests for PointCloud data structure and transformations."""

import numpy as np
import pytest

from module_09_3d.point_cloud import PointCloud


class TestPointCloud:
    def test_valid_point_cloud_creation(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        confs = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([1, 2], dtype=np.int32)

        pc = PointCloud(points=pts, confidences=confs, semantic_classes=classes)
        assert pc.num_points == 2
        assert not pc.is_empty()

    def test_rejects_non_finite_points(self):
        pts_nan = np.array([[1.0, np.nan, 3.0]])
        with pytest.raises(ValueError, match="non-finite"):
            PointCloud(points=pts_nan)

        pts_inf = np.array([[1.0, np.inf, 3.0]])
        with pytest.raises(ValueError, match="non-finite"):
            PointCloud(points=pts_inf)

    def test_rejects_malformed_shapes(self):
        pts_1d = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="shape"):
            PointCloud(points=pts_1d)

        pts_2d_bad_cols = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="shape"):
            PointCloud(points=pts_2d_bad_cols)

    def test_rejects_invalid_confidence_range(self):
        pts = np.zeros((2, 3))
        with pytest.raises(ValueError, match="range"):
            PointCloud(points=pts, confidences=np.array([1.5, 0.5]))

    def test_bounding_box_and_center(self):
        pts = np.array([[-1.0, 0.0, 2.0], [3.0, 4.0, -2.0]])
        pc = PointCloud(points=pts)

        min_b, max_b = pc.bounding_box()
        assert np.allclose(min_b, np.array([-1.0, 0.0, -2.0]))
        assert np.allclose(max_b, np.array([3.0, 4.0, 2.0]))

        center = pc.center()
        assert np.allclose(center, np.array([1.0, 2.0, 0.0]))

    def test_translation_and_scaling(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        pc = PointCloud(points=pts)

        pc_trans = pc.translate([1.0, -1.0, 0.0])
        assert np.allclose(pc_trans.points, np.array([[2.0, 1.0, 3.0]]))
        assert np.allclose(pc.points, np.array([[1.0, 2.0, 3.0]]))  # Original uncorrupted

        pc_scaled = pc.scale(2.0)
        assert np.allclose(pc_scaled.points, np.array([[2.0, 4.0, 6.0]]))

    def test_filter_by_confidence(self):
        pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        confs = np.array([0.2, 0.7, 0.9], dtype=np.float32)
        pc = PointCloud(points=pts, confidences=confs)

        filtered = pc.filter_by_confidence(0.5)
        assert filtered.num_points == 2
        assert np.allclose(filtered.points, np.array([[1, 1, 1], [2, 2, 2]]))
