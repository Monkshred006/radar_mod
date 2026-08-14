"""Module 3: Sensor Fusion + Feature Extraction Package."""

from module_03_sensor_fusion.config import (
    Module3Config,
    OpticalFeatureConfig,
    EnvironmentalFeatureConfig,
    MotionFeatureConfig,
    DistanceFeatureConfig,
    QualityFeatureConfig,
    TokenizerConfig,
)
from module_03_sensor_fusion.sensor_groups import SensorGroupRegistry, DEFAULT_SENSOR_GROUPS
from module_03_sensor_fusion.feature_extractors import (
    compute_first_difference,
    compute_rate_of_change,
    compute_rolling_mean,
    compute_rolling_std,
    compute_rolling_energy,
)
from module_03_sensor_fusion.optical_features import extract_optical_features
from module_03_sensor_fusion.environmental_features import extract_environmental_features
from module_03_sensor_fusion.motion_features import extract_motion_features
from module_03_sensor_fusion.distance_features import extract_distance_features
from module_03_sensor_fusion.quality_features import extract_quality_features
from module_03_sensor_fusion.fusion import fuse_sensor_features
from module_03_sensor_fusion.tokenization import SensorAwareTokenizer
from module_03_sensor_fusion.pipeline import SensorFusionPipeline, FusionOutput

__all__ = [
    "Module3Config",
    "OpticalFeatureConfig",
    "EnvironmentalFeatureConfig",
    "MotionFeatureConfig",
    "DistanceFeatureConfig",
    "QualityFeatureConfig",
    "TokenizerConfig",
    "SensorGroupRegistry",
    "DEFAULT_SENSOR_GROUPS",
    "compute_first_difference",
    "compute_rate_of_change",
    "compute_rolling_mean",
    "compute_rolling_std",
    "compute_rolling_energy",
    "extract_optical_features",
    "extract_environmental_features",
    "extract_motion_features",
    "extract_distance_features",
    "extract_quality_features",
    "fuse_sensor_features",
    "SensorAwareTokenizer",
    "SensorFusionPipeline",
    "FusionOutput",
]
