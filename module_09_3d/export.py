"""Export utilities for 3D point clouds, scenes, and rendered frame sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from module_09_3d.point_cloud import PointCloud
from module_09_3d.scene import Scene3D


def export_point_cloud_ply(point_cloud: PointCloud, file_path: Union[str, Path]) -> Path:
    """Export PointCloud to ASCII Stanford .ply format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = point_cloud.points
    n_pts = len(pts)

    header = [
        "ply",
        "format ascii 1.0",
        "comment Created by PhotonShield AI Module 9",
        f"element vertex {n_pts}",
        "property float x",
        "property float y",
        "property float z",
    ]

    has_conf = point_cloud.confidences is not None
    if has_conf:
        header.append("property float confidence")

    has_class = point_cloud.semantic_classes is not None
    if has_class:
        header.append("property int class_id")

    header.extend(["end_header", ""])

    lines = ["\n".join(header)]

    for i in range(n_pts):
        line = f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f}"
        if has_conf:
            line += f" {point_cloud.confidences[i]:.4f}"
        if has_class:
            line += f" {int(point_cloud.semantic_classes[i])}"
        lines.append(line)

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


def export_point_cloud_npy(point_cloud: PointCloud, file_path: Union[str, Path]) -> Path:
    """Export PointCloud points to binary .npy format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, point_cloud.points)
    return path


def export_scene_metadata_json(scene: Scene3D, file_path: Union[str, Path]) -> Path:
    """Export Scene3D metadata and object list to .json format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scene.to_dict(), f, indent=2)
    return path


def export_frame_sequence_png(
    frames: List[np.ndarray],
    output_dir: Union[str, Path],
    prefix: str = "frame",
) -> List[Path]:
    """Export list of RGB image frames as PNG files."""
    try:
        from PIL import Image
    except ImportError:
        Image = None

    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for i, frame in enumerate(frames):
        filename = dir_path / f"{prefix}_{i:03d}.png"
        if Image is not None:
            img = Image.fromarray(frame)
            img.save(filename)
        else:
            # Fallback save raw numpy array if PIL not installed
            filename = dir_path / f"{prefix}_{i:03d}.npy"
            np.save(filename, frame)
        saved_paths.append(filename)

    return saved_paths
