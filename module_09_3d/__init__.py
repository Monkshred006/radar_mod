"""Module 9: 3D Reconstruction Interface & Visualization Pipeline for PhotonShield AI.

Provides:
- 3D PointCloud and Scene3D representations
- ThreeDReconstructor plug-in interface and synthetic geometry generators
- VirtualCamera, 3D-to-2D projection math, and PointCloudRenderer
- 360-degree RotatingViewGenerator
- FrameStream buffered streaming engine
- Simulated and Hardware OLED display backends
- TrapezoidalPrismRenderer (Pepper's-Ghost pseudo-holographic multi-view)
- High-level PhotonShield3DPipeline
- Stanford .ply, .npy, and .json export utilities
"""

from module_09_3d.camera import VirtualCamera
from module_09_3d.config import (
    CameraConfig,
    ExportConfig,
    Module9Config,
    OLEDConfig,
    PointCloudConfig,
    PrismConfig,
    ReconstructionConfig,
    RenderConfig,
    RotationConfig,
    SceneConfig,
)
from module_09_3d.display import FrameStream
from module_09_3d.export import (
    export_frame_sequence_png,
    export_point_cloud_npy,
    export_point_cloud_ply,
    export_scene_metadata_json,
)
from module_09_3d.frame_generator import RotatingViewGenerator
from module_09_3d.inference import PhotonShield3DPipeline
from module_09_3d.interfaces import (
    DisplayBackend,
    ReconstructionEvaluator,
    SceneInput,
    ThreeDReconstructor,
)
from module_09_3d.oled import (
    HardwareDisplayBackend,
    SimulatedDisplayBackend,
    build_display_backend,
)
from module_09_3d.point_cloud import PointCloud
from module_09_3d.prism_display import TrapezoidalPrismRenderer
from module_09_3d.projection import (
    compute_look_at_matrix,
    compute_orthographic_matrix,
    compute_perspective_matrix,
    project_points,
)
from module_09_3d.reconstruction import (
    PassThroughReconstructor,
    SyntheticGeometryReconstructor,
    build_reconstructor,
    generate_synthetic_cube,
    generate_synthetic_sphere,
    generate_synthetic_vehicle,
)
from module_09_3d.renderer import PointCloudRenderer
from module_09_3d.rotation import RotationTrajectory
from module_09_3d.scene import Object3D, Scene3D

__all__ = [
    # Configurations
    "PointCloudConfig",
    "SceneConfig",
    "ReconstructionConfig",
    "CameraConfig",
    "RenderConfig",
    "RotationConfig",
    "OLEDConfig",
    "PrismConfig",
    "ExportConfig",
    "Module9Config",
    # Core 3D Data
    "PointCloud",
    "Scene3D",
    "Object3D",
    "SceneInput",
    # Interfaces
    "ThreeDReconstructor",
    "DisplayBackend",
    "ReconstructionEvaluator",
    # Reconstruction Backends
    "SyntheticGeometryReconstructor",
    "PassThroughReconstructor",
    "build_reconstructor",
    "generate_synthetic_cube",
    "generate_synthetic_sphere",
    "generate_synthetic_vehicle",
    # Math & Camera
    "compute_look_at_matrix",
    "compute_perspective_matrix",
    "compute_orthographic_matrix",
    "project_points",
    "VirtualCamera",
    # Rendering & Rotation
    "PointCloudRenderer",
    "RotationTrajectory",
    "RotatingViewGenerator",
    "TrapezoidalPrismRenderer",
    # Display & Streaming
    "FrameStream",
    "SimulatedDisplayBackend",
    "HardwareDisplayBackend",
    "build_display_backend",
    # Pipeline & Exporters
    "PhotonShield3DPipeline",
    "export_point_cloud_ply",
    "export_point_cloud_npy",
    "export_scene_metadata_json",
    "export_frame_sequence_png",
]
