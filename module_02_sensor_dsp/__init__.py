"""Module 2: Sensor Signal Preprocessing / DSP — Package."""

from module_02_sensor_dsp.config import (
    SensorDSPConfig,
    FilterConfig,
    BaselineConfig,
    OutlierConfig,
    NormalizationConfig,
    MissingDataConfig,
    SyncConfig,
    ChannelConfig,
    PHOTONSHIELD_CHANNELS,
)
from module_02_sensor_dsp.filters import apply_filter
from module_02_sensor_dsp.denoising import apply_baseline_correction
from module_02_sensor_dsp.outliers import detect_outliers
from module_02_sensor_dsp.normalization import SensorScaler
from module_02_sensor_dsp.synchronization import (
    synchronize_all_channels_offline,
    synchronize_channel_streaming,
    build_target_grid,
)
from module_02_sensor_dsp.quality import compute_channel_quality, compute_all_quality
from module_02_sensor_dsp.pipeline import SensorDSPPipeline, RawSensorData, ProcessedOutput

__all__ = [
    # Config
    "SensorDSPConfig",
    "FilterConfig",
    "BaselineConfig",
    "OutlierConfig",
    "NormalizationConfig",
    "MissingDataConfig",
    "SyncConfig",
    "ChannelConfig",
    "PHOTONSHIELD_CHANNELS",
    # Filters
    "apply_filter",
    # Denoising
    "apply_baseline_correction",
    # Outliers
    "detect_outliers",
    # Normalization
    "SensorScaler",
    # Sync
    "synchronize_all_channels_offline",
    "synchronize_channel_streaming",
    "build_target_grid",
    # Quality
    "compute_channel_quality",
    "compute_all_quality",
    # Pipeline
    "SensorDSPPipeline",
    "RawSensorData",
    "ProcessedOutput",
]
