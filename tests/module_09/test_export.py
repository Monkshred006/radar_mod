"""Tests for 3D export utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from module_09_3d.export import (
    export_frame_sequence_png,
    export_point_cloud_npy,
    export_point_cloud_ply,
    export_scene_metadata_json,
)
from module_09_3d.point_cloud import PointCloud
from module_09_3d.scene import Object3D, Scene3D


class TestExport:
    def test_export_ply_roundtrip_structure(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        confs = np.array([0.95, 0.85], dtype=np.float32)
        classes = np.array([1, 2], dtype=np.int32)
        pc = PointCloud(points=pts, confidences=confs, semantic_classes=classes)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test.ply"
            res = export_point_cloud_ply(pc, out_file)
            assert res.exists()

            content = out_file.read_text(encoding="utf-8")
            assert "element vertex 2" in content
            assert "property float x" in content
            assert "property float confidence" in content
            assert "1.000000 2.000000 3.000000 0.9500 1" in content

    def test_export_npy(self):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud(points=pts)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test.npy"
            export_point_cloud_npy(pc, out_file)
            assert out_file.exists()

            loaded = np.load(out_file)
            assert np.allclose(loaded, pts)

    def test_export_json_metadata(self):
        pc = PointCloud(points=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
        scene = Scene3D(point_cloud=pc, timestamp=12.34)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "scene.json"
            export_scene_metadata_json(scene, out_file)
            assert out_file.exists()

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["num_points"] == 2
            assert data["timestamp"] == 12.34

    def test_export_png_sequence(self):
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_frame_sequence_png(frames, tmpdir, prefix="test_frame")
            assert len(paths) == 3
            for p in paths:
                assert p.exists()
