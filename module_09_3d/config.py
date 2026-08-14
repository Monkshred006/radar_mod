"""Configuration dataclasses for Module 9 — 3D Reconstruction Interface & Visualization Pipeline.

Provides modular configurations for:
- PointCloudConfig: coordinate conventions, validation thresholds, attribute schemas.
- SceneConfig: scene metadata, bounding box parameters.
- ReconstructionConfig: 3D reconstruction backend selection and hyperparameters.
- CameraConfig: virtual camera parameters (FOV, look-at, resolution, near/far clip).
- RenderConfig: visual style, point size, color maps, background color.
- RotationConfig: 360° rotation angles, angular step, FPS, frame counts.
- OLEDConfig: display resolution, timing, brightness, simulated canvas settings.
- PrismConfig: pseudo-holographic trapezoidal prism dimensions, 4-view offsets, Pepper's-ghost layout.
- ExportConfig: export directories, formats (.ply, .npy, .json, .png sequence).
- Module9Config: root container aggregating all sub-configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


@dataclass
class PointCloudConfig:
    """Configures point-cloud data structure rules and validation."""

    coordinate_system: str = "right_handed_z_up"  # x=lateral, y=depth/range, z=vertical
    max_points: int = 100_000
    validate_finite: bool = True
    normalize_bounding_box: bool = False
    confidence_min: float = 0.0
    confidence_max: float = 1.0


@dataclass
class SceneConfig:
    """Configures 3D scene properties."""

    coordinate_frame: str = "sensor_local"  # "sensor_local", "world", "enu"
    unit: str = "meters"
    enable_bounding_box: bool = True
    default_bounding_box_padding: float = 0.1


@dataclass
class ReconstructionConfig:
    """Configures 3D reconstruction backend.

    backend choices:
    - "synthetic_geometry": Demo cube, sphere, vehicle point-clouds (SOFTWARE VERIFICATION ONLY).
    - "passthrough": Directly consumes pre-constructed PointCloud from input.
    - "learned_radar_to_3d": Reserved for future neural reconstruction model.
    """

    backend: Literal["synthetic_geometry", "passthrough", "learned_radar_to_3d"] = "synthetic_geometry"
    synthetic_geometry_type: Literal["cube", "sphere", "vehicle", "multi_object"] = "vehicle"
    num_synthetic_points: int = 500
    noise_std: float = 0.02


@dataclass
class CameraConfig:
    """Virtual camera parameters for 3D projection."""

    projection_type: Literal["perspective", "orthographic"] = "perspective"
    fov_degrees: float = 45.0
    near_clip: float = 0.1
    far_clip: float = 100.0
    camera_position: List[float] = field(default_factory=lambda: [0.0, -3.5, 1.5])
    camera_target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    camera_up: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    image_width: int = 256
    image_height: int = 256


@dataclass
class RenderConfig:
    """3D point-cloud rendering styles."""

    point_size: int = 3
    color_mode: Literal["solid", "depth", "semantic_class", "confidence", "velocity"] = "depth"
    solid_color: Tuple[int, int, int] = (0, 255, 200)  # Cyan/teal default
    background_color: Tuple[int, int, int] = (0, 0, 0)   # Black OLED background
    depth_colormap: str = "viridis"
    class_colormap: Dict[int, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            0: (128, 128, 128),  # background/unknown
            1: (0, 255, 100),    # target/vehicle
            2: (255, 50, 50),    # anomaly/hazard
            3: (50, 150, 255),   # environment/clutter
        }
    )


@dataclass
class RotationConfig:
    """Configures 360° rotating view generation.

    Generates angles from start_deg to end_deg (exclusive of end_deg if end-start=360)
    with step_deg.
    """

    rotation_start_deg: float = 0.0
    rotation_end_deg: float = 360.0
    rotation_step_deg: float = 15.0  # 360/15 = 24 unique frames
    elevation_deg: float = 20.0
    distance: float = 3.5
    fps: int = 24
    clockwise: bool = False

    @property
    def num_frames(self) -> int:
        """Calculate unique frames without repeating 0° and 360°."""
        span = abs(self.rotation_end_deg - self.rotation_start_deg)
        if span >= 360.0:
            return int(round(360.0 / self.rotation_step_deg))
        return int(round(span / self.rotation_step_deg))


@dataclass
class OLEDConfig:
    """OLED display interface parameters."""

    backend_type: Literal["simulated", "hardware_interface"] = "simulated"
    display_width: int = 256
    display_height: int = 256
    color_depth_bits: int = 24
    target_fps: int = 24
    brightness: float = 1.0  # [0.0, 1.0]
    buffer_max_size: int = 120
    drop_frames_on_overflow: bool = True
    loop_playback: bool = True


@dataclass
class PrismConfig:
    """Configures transparent trapezoidal prism / pseudo-holographic Pepper's-Ghost visualization.

    Multi-view layout:
    - 4 synchronized orthogonal views (Front, Back, Left, Right) positioned around a central cross.
    """

    enable_prism_mode: bool = False
    prism_shape: Literal["four_face_trapezoid", "single_face"] = "four_face_trapezoid"
    canvas_width: int = 512
    canvas_height: int = 512
    view_scale: float = 0.35  # Sub-view scale relative to canvas
    optical_tilt_deg: float = 45.0
    mirror_x: bool = False
    mirror_y: bool = False


@dataclass
class ExportConfig:
    """Export parameters for 3D point-clouds and rendered sequences."""

    output_dir: str = "exports_3d"
    export_ply: bool = True
    export_npy: bool = True
    export_json_metadata: bool = True
    export_png_sequence: bool = True


@dataclass
class Module9Config:
    """Root configuration for Module 9."""

    point_cloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    oled: OLEDConfig = field(default_factory=OLEDConfig)
    prism: PrismConfig = field(default_factory=PrismConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    seed: int = 42
