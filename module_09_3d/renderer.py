"""Lightweight point cloud rasterizer / renderer for Module 9."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from module_09_3d.camera import VirtualCamera
from module_09_3d.config import RenderConfig
from module_09_3d.point_cloud import PointCloud
from module_09_3d.scene import Scene3D


def _get_colormap_rgb(values_norm: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """Generate RGB colors [0..255] from normalized float values in [0, 1]."""
    # Lightweight pure-numpy gradient colormap
    v = np.clip(values_norm, 0.0, 1.0)
    if colormap == "viridis":
        # Approximate viridis: purple (68, 1, 84) -> teal (33, 145, 140) -> yellow (253, 231, 37)
        r = (68 + (253 - 68) * v).astype(np.uint8)
        g = (1 + (231 - 1) * (v ** 0.8)).astype(np.uint8)
        b = (84 + (37 - 84) * v + 120 * np.sin(v * np.pi)).astype(np.uint8)
        return np.column_stack([r, g, b])
    elif colormap == "plasma":
        r = (13 + 240 * v).astype(np.uint8)
        g = (8 + 180 * (v ** 1.2)).astype(np.uint8)
        b = (135 + 100 * np.cos(v * np.pi)).astype(np.uint8)
        return np.column_stack([r, g, b])
    else:
        # Cyan-blue to yellow gradient
        r = (255 * v).astype(np.uint8)
        g = (255 * (1.0 - np.abs(v - 0.5) * 2)).astype(np.uint8)
        b = (255 * (1.0 - v)).astype(np.uint8)
        return np.column_stack([r, g, b])


class PointCloudRenderer:
    """Renders 3D PointClouds / Scene3D objects into 2D RGB images with depth buffering."""

    def __init__(
        self,
        camera: Optional[VirtualCamera] = None,
        config: Optional[RenderConfig] = None,
    ) -> None:
        self.camera = camera or VirtualCamera()
        self.config = config or RenderConfig()

    def render(self, scene_or_pc: PointCloud | Scene3D) -> np.ndarray:
        """Render point cloud into an (H, W, 3) uint8 image with z-buffering.

        Returns
        -------
        image : ndarray[H, W, 3], uint8
            RGB rendered frame.
        """
        if isinstance(scene_or_pc, Scene3D):
            pc = scene_or_pc.point_cloud
        else:
            pc = scene_or_pc

        w = self.camera.config.image_width
        h = self.camera.config.image_height

        # Initialize image and depth buffer
        bg = np.array(self.config.background_color, dtype=np.uint8)
        image = np.full((h, w, 3), bg, dtype=np.uint8)
        depth_buffer = np.full((h, w), np.inf, dtype=np.float32)

        if pc.is_empty():
            return image

        # 1. Project points
        coords, depths, valid = self.camera.project(pc.points)
        if not valid.any():
            return image

        coords_v = coords[valid]
        depths_v = depths[valid]
        n_visible = len(coords_v)

        # 2. Determine point colors
        colors = np.zeros((n_visible, 3), dtype=np.uint8)
        mode = self.config.color_mode

        if mode == "solid":
            colors[:] = self.config.solid_color
        elif mode == "depth":
            d_min, d_max = depths_v.min(), depths_v.max()
            if d_max > d_min:
                d_norm = (depths_v - d_min) / (d_max - d_min)
            else:
                d_norm = np.zeros(n_visible, dtype=np.float32)
            colors = _get_colormap_rgb(d_norm, self.config.depth_colormap)
        elif mode == "semantic_class" and pc.semantic_classes is not None:
            classes = pc.semantic_classes[valid]
            cmap = self.config.class_colormap
            for i, cls in enumerate(classes):
                colors[i] = cmap.get(int(cls), (200, 200, 200))
        elif mode == "confidence" and pc.confidences is not None:
            confs = pc.confidences[valid]
            colors = _get_colormap_rgb(confs, "plasma")
        else:
            colors[:] = self.config.solid_color

        # 3. Rasterize with depth buffer and point splatting
        radius = max(self.config.point_size // 2, 0)

        for (u, v), depth, color in zip(coords_v, depths_v, colors):
            u_min = max(0, u - radius)
            u_max = min(w, u + radius + 1)
            v_min = max(0, v - radius)
            v_max = min(h, v + radius + 1)

            for py in range(v_min, v_max):
                for px in range(u_min, u_max):
                    if depth < depth_buffer[py, px]:
                        depth_buffer[py, px] = depth
                        image[py, px] = color

        return image
