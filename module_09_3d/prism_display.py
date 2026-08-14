"""Trapezoidal prism multi-view renderer for pseudo-holographic Pepper's-Ghost display."""

from __future__ import annotations

from typing import Optional

import numpy as np

from module_09_3d.camera import VirtualCamera
from module_09_3d.config import CameraConfig, PrismConfig, RenderConfig
from module_09_3d.point_cloud import PointCloud
from module_09_3d.renderer import PointCloudRenderer
from module_09_3d.scene import Scene3D


class TrapezoidalPrismRenderer:
    """Renders 4 synchronized orthogonal views onto a single canvas for a 4-face transparent prism.

    Layout (Pepper's-Ghost 4-face pseudo-hologram):
                       [Top View / Back]
                               ▲
        [Left View] ◄   [Center Black]   ► [Right View]
                               ▼
                      [Bottom View / Front]

    When displayed on an OLED placed below a 4-face transparent trapezoidal prism,
    each face reflects its corresponding sub-image upwards, creating a floating
    pseudo-holographic 3D visualization.

    DISCLAIMER
    ----------
    This system produces a pseudo-holographic / Pepper's-Ghost multi-view optical
    visualization, NOT a true volumetric hologram. Optical calibration depends on
    physical prism angles and glass thickness.
    """

    def __init__(
        self,
        prism_config: Optional[PrismConfig] = None,
        camera_config: Optional[CameraConfig] = None,
        render_config: Optional[RenderConfig] = None,
    ) -> None:
        self.prism_config = prism_config or PrismConfig()
        self.camera_config = camera_config or CameraConfig()
        self.render_config = render_config or RenderConfig()

        self.camera = VirtualCamera(self.camera_config)
        self.renderer = PointCloudRenderer(self.camera, self.render_config)

    def render_prism_canvas(self, scene_or_pc: Scene3D | PointCloud) -> np.ndarray:
        """Render a full 4-face composite canvas.

        Returns
        -------
        canvas : ndarray[canvas_height, canvas_width, 3], uint8
        """
        cw = self.prism_config.canvas_width
        ch = self.prism_config.canvas_height
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)

        if isinstance(scene_or_pc, Scene3D):
            target = scene_or_pc.point_cloud.center()
        else:
            target = scene_or_pc.center()

        self.camera.set_target(target)
        dist = np.linalg.norm(np.array(self.camera_config.camera_position) - target)
        dist = max(dist, 1.0)
        elev = np.array(self.camera_config.camera_position)[2]

        # 4 viewing directions: Front (0°), Right (90°), Back (180°), Left (270°)
        views = {
            "front": (0.0, np.array([target[0], target[1] - dist, elev], dtype=np.float32)),
            "right": (90.0, np.array([target[0] + dist, target[1], elev], dtype=np.float32)),
            "back": (180.0, np.array([target[0], target[1] + dist, elev], dtype=np.float32)),
            "left": (270.0, np.array([target[0] - dist, target[1], elev], dtype=np.float32)),
        }

        # Sub-view dimensions on canvas
        sub_w = int(cw * self.prism_config.view_scale)
        sub_h = int(ch * self.prism_config.view_scale)

        # Temporary camera config with sub-view resolution
        self.camera.config.image_width = sub_w
        self.camera.config.image_height = sub_h

        # 1. Front View (Bottom of canvas, right-side up)
        self.camera.set_position(views["front"][1])
        img_front = self.renderer.render(scene_or_pc)
        bx = (cw - sub_w) // 2
        by = ch - sub_h
        canvas[by:by + sub_h, bx:bx + sub_w] = img_front

        # 2. Back View (Top of canvas, upside down for Pepper's ghost reflection)
        self.camera.set_position(views["back"][1])
        img_back = np.flipud(np.fliplr(self.renderer.render(scene_or_pc)))
        tx = (cw - sub_w) // 2
        ty = 0
        canvas[ty:ty + sub_h, tx:tx + sub_w] = img_back

        # 3. Left View (Left side of canvas, rotated 90° clockwise)
        self.camera.set_position(views["left"][1])
        img_left = np.rot90(self.renderer.render(scene_or_pc), k=-1)
        lx = 0
        ly = (ch - img_left.shape[0]) // 2
        canvas[ly:ly + img_left.shape[0], lx:lx + img_left.shape[1]] = img_left

        # 4. Right View (Right side of canvas, rotated 90° counter-clockwise)
        self.camera.set_position(views["right"][1])
        img_right = np.rot90(self.renderer.render(scene_or_pc), k=1)
        rx = cw - img_right.shape[1]
        ry = (ch - img_right.shape[0]) // 2
        canvas[ry:ry + img_right.shape[0], rx:rx + img_right.shape[1]] = img_right

        # Restore camera config dimensions
        self.camera.config.image_width = self.camera_config.image_width
        self.camera.config.image_height = self.camera_config.image_height

        return canvas
