"""High-level 3D Reconstruction & Visualization pipeline for Module 9."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from module_09_3d.config import Module9Config
from module_09_3d.display import FrameStream
from module_09_3d.frame_generator import RotatingViewGenerator
from module_09_3d.interfaces import DisplayBackend, SceneInput, ThreeDReconstructor
from module_09_3d.oled import SimulatedDisplayBackend, build_display_backend
from module_09_3d.point_cloud import PointCloud
from module_09_3d.prism_display import TrapezoidalPrismRenderer
from module_09_3d.reconstruction import build_reconstructor
from module_09_3d.renderer import PointCloudRenderer
from module_09_3d.scene import Scene3D


class PhotonShield3DPipeline:
    """High-level pipeline connecting 3D Reconstruction, 360 Rendering, and Display Backends."""

    def __init__(self, config: Optional[Module9Config] = None) -> None:
        self.config = config or Module9Config()
        self.reconstructor: ThreeDReconstructor = build_reconstructor(self.config.reconstruction)
        self.rotation_generator = RotatingViewGenerator(
            rotation_config=self.config.rotation,
            camera_config=self.config.camera,
            render_config=self.config.render,
        )
        self.prism_renderer = TrapezoidalPrismRenderer(
            prism_config=self.config.prism,
            camera_config=self.config.camera,
            render_config=self.config.render,
        )
        self.display_backend = build_display_backend(self.config.oled)
        self.frame_stream = FrameStream(self.config.oled)

    def reconstruct(self, scene_input: SceneInput) -> Scene3D:
        """Reconstruct 3D scene from upstream sensor/state representations."""
        return self.reconstructor.reconstruct(scene_input)

    def render_rotating_views(self, scene: Scene3D | PointCloud) -> List[np.ndarray]:
        """Generate ordered 360-degree rotating frames without mutating the 3D scene."""
        return self.rotation_generator.generate_frames(scene)

    def render_prism_view(self, scene: Scene3D | PointCloud) -> np.ndarray:
        """Render 4-face pseudo-holographic composite canvas for trapezoidal prism."""
        return self.prism_renderer.render_prism_canvas(scene)

    def reconstruct_and_render(
        self,
        scene_input: SceneInput,
    ) -> Tuple[Scene3D, List[np.ndarray]]:
        """End-to-end processing: SceneInput -> Scene3D -> 360° Rotating 2D Views."""
        scene = self.reconstruct(scene_input)
        frames = self.render_rotating_views(scene)
        return scene, frames

    def stream_frames_to_display(
        self,
        frames: List[np.ndarray],
        display: Optional[DisplayBackend] = None,
    ) -> int:
        """Push frame sequence to display backend. Returns count of displayed frames."""
        disp = display or self.display_backend
        displayed = 0
        for f in frames:
            if disp.display_frame(f):
                displayed += 1
        return displayed
