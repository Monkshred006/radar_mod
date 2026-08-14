"""Configuration for Module 2 — Sensor Signal Preprocessing / DSP.

Supported PhotonShield sensor channels:
- photodiode_1, photodiode_2  (BPW34 optical sensing)
- temperature, humidity, pressure  (BME280)
- accel_x, accel_y, accel_z  (MPU6050 accelerometer)
- gyro_x, gyro_y, gyro_z  (MPU6050 gyroscope)
- distance  (VL53L0X)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
import yaml


# ---------------------------------------------------------------------------
# Filter configuration
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    """Per-channel filter configuration.

    Attributes:
        filter_type: One of "moving_average", "ema", "median", "lowpass", "none".
        window: Window length for moving_average or median (number of samples).
        alpha: Smoothing factor for EMA in [0,1]. Higher = less smoothing.
        cutoff_hz: Cutoff frequency for lowpass filter (Hz). Requires sampling_rate.
        sampling_rate_hz: Sampling rate of the signal (Hz). Required for lowpass.
        order: Filter order for Butterworth lowpass.
    """
    filter_type: Literal["moving_average", "ema", "median", "lowpass", "none"] = "none"
    window: int = 5
    alpha: float = 0.1
    cutoff_hz: float = 1.0
    sampling_rate_hz: float = 10.0
    order: int = 2


@dataclass
class BaselineConfig:
    """Baseline correction settings (primarily for photodiode channels).

    Attributes:
        enabled: Whether to apply baseline correction.
        method: "ema" uses a slow EMA track as baseline; "percentile" uses a
                rolling percentile.
        alpha: EMA smoothing alpha for baseline tracker (very small → slow track).
        percentile: Percentile for rolling baseline estimate (e.g. 5 → 5th pct).
        window: Window for rolling percentile method.
    """
    enabled: bool = False
    method: Literal["ema", "percentile"] = "ema"
    alpha: float = 0.005
    percentile: float = 5.0
    window: int = 50


# ---------------------------------------------------------------------------
# Outlier configuration
# ---------------------------------------------------------------------------

@dataclass
class OutlierConfig:
    """Outlier detection configuration.

    Attributes:
        method: Detection method — "iqr", "zscore", "mad", "range", "none".
        threshold: Multiplier for IQR/MAD or sigma cutoff for z-score.
        min_val: Physical minimum valid value (used with "range").
        max_val: Physical maximum valid value (used with "range").
    """
    method: Literal["iqr", "zscore", "mad", "range", "none"] = "none"
    threshold: float = 3.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None


# ---------------------------------------------------------------------------
# Normalization configuration
# ---------------------------------------------------------------------------

@dataclass
class NormalizationConfig:
    """Normalization/scaling settings.

    Attributes:
        method: "minmax", "standard", "robust", "none".
        stats_path: Path to JSON file holding pre-fitted statistics
                    (loaded for val/test; saved after fitting on train).
        feature_range: Output range for minmax scaling [lo, hi].
        clip: Whether to clip values to feature_range during transform.
    """
    method: Literal["minmax", "standard", "robust", "none"] = "none"
    stats_path: Optional[str] = None
    feature_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    clip: bool = False


# ---------------------------------------------------------------------------
# Missing data configuration
# ---------------------------------------------------------------------------

@dataclass
class MissingDataConfig:
    """Strategy for handling missing / NaN values.

    Attributes:
        strategy: "leave", "ffill" (forward-fill / zero-order hold),
                  "interpolate" (linear, offline only), "constant".
        fill_value: Value used when strategy is "constant".
    """
    strategy: Literal["leave", "ffill", "interpolate", "constant"] = "ffill"
    fill_value: float = 0.0


# ---------------------------------------------------------------------------
# Synchronization configuration
# ---------------------------------------------------------------------------

@dataclass
class SyncConfig:
    """Temporal synchronization settings.

    Attributes:
        target_rate_hz: Output sampling rate. All channels are resampled here.
        method_offline: Resampling method for offline mode:
                        "nearest", "ffill", "linear".
        method_streaming: Resampling method for streaming mode.
                          MUST be causal — only "nearest" or "ffill" allowed.
        max_gap_s: Maximum gap (seconds) across which interpolation/ffill is
                   allowed. Larger gaps yield NaN (flagged as missing).
    """
    target_rate_hz: float = 10.0
    method_offline: Literal["nearest", "ffill", "linear"] = "linear"
    method_streaming: Literal["nearest", "ffill"] = "ffill"
    max_gap_s: float = 1.0


# ---------------------------------------------------------------------------
# Per-sensor channel config
# ---------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    """Preprocessing configuration for a single sensor channel."""
    filter: FilterConfig = field(default_factory=FilterConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    missing: MissingDataConfig = field(default_factory=MissingDataConfig)


# ---------------------------------------------------------------------------
# Top-level DSP configuration
# ---------------------------------------------------------------------------

PHOTONSHIELD_CHANNELS = [
    "photodiode_1",
    "photodiode_2",
    "temperature",
    "humidity",
    "pressure",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "distance",
]


def _default_channel_configs() -> Dict[str, ChannelConfig]:
    """Build sensible default channel configurations for PhotonShield sensors."""
    configs: Dict[str, ChannelConfig] = {}

    # Photodiodes — low-pass + baseline removal
    for ch in ("photodiode_1", "photodiode_2"):
        configs[ch] = ChannelConfig(
            filter=FilterConfig(filter_type="lowpass", cutoff_hz=5.0,
                                sampling_rate_hz=100.0, order=2),
            baseline=BaselineConfig(enabled=True, method="ema", alpha=0.005),
            outlier=OutlierConfig(method="mad", threshold=5.0),
            normalization=NormalizationConfig(method="minmax"),
            missing=MissingDataConfig(strategy="ffill"),
        )

    # Environmental sensors — light smoothing, outlier via physical range
    configs["temperature"] = ChannelConfig(
        filter=FilterConfig(filter_type="ema", alpha=0.2),
        outlier=OutlierConfig(method="range", min_val=-40.0, max_val=85.0),
        normalization=NormalizationConfig(method="standard"),
        missing=MissingDataConfig(strategy="ffill"),
    )
    configs["humidity"] = ChannelConfig(
        filter=FilterConfig(filter_type="ema", alpha=0.2),
        outlier=OutlierConfig(method="range", min_val=0.0, max_val=100.0),
        normalization=NormalizationConfig(method="standard"),
        missing=MissingDataConfig(strategy="ffill"),
    )
    configs["pressure"] = ChannelConfig(
        filter=FilterConfig(filter_type="ema", alpha=0.1),
        outlier=OutlierConfig(method="range", min_val=300.0, max_val=1100.0),
        normalization=NormalizationConfig(method="standard"),
        missing=MissingDataConfig(strategy="ffill"),
    )

    # IMU channels — median smoothing, z-score outlier
    for ch in ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"):
        configs[ch] = ChannelConfig(
            filter=FilterConfig(filter_type="ema", alpha=0.3),
            outlier=OutlierConfig(method="zscore", threshold=4.0),
            normalization=NormalizationConfig(method="standard"),
            missing=MissingDataConfig(strategy="ffill"),
        )

    # Distance — median filter, range check
    configs["distance"] = ChannelConfig(
        filter=FilterConfig(filter_type="median", window=5),
        outlier=OutlierConfig(method="range", min_val=0.0, max_val=2000.0),
        normalization=NormalizationConfig(method="minmax"),
        missing=MissingDataConfig(strategy="ffill"),
    )

    return configs


@dataclass
class SensorDSPConfig:
    """Top-level configuration for the Sensor DSP Pipeline.

    Attributes:
        sync: Synchronization settings.
        channels: Per-channel ChannelConfig map.
        precision: Numerical precision for DSP operations ("float32" or "float64").
        streaming: If True, pipeline enforces causal-only operations.
        random_seed: Seed for reproducible operations.
    """
    sync: SyncConfig = field(default_factory=SyncConfig)
    channels: Dict[str, ChannelConfig] = field(default_factory=_default_channel_configs)
    precision: Literal["float32", "float64"] = "float64"
    streaming: bool = False
    random_seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_yaml(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict) -> "SensorDSPConfig":
        """Reconstruct config from a dictionary (e.g. loaded from YAML)."""
        # Rebuild nested dataclasses
        sync_data = data.pop("sync", {})
        channels_data = data.pop("channels", {})
        config = cls(**data)
        if sync_data:
            config.sync = SyncConfig(**sync_data)
        if channels_data:
            config.channels = {}
            for name, ch_data in channels_data.items():
                f_data = ch_data.pop("filter", {})
                b_data = ch_data.pop("baseline", {})
                o_data = ch_data.pop("outlier", {})
                n_data = ch_data.pop("normalization", {})
                m_data = ch_data.pop("missing", {})
                config.channels[name] = ChannelConfig(
                    filter=FilterConfig(**f_data) if f_data else FilterConfig(),
                    baseline=BaselineConfig(**b_data) if b_data else BaselineConfig(),
                    outlier=OutlierConfig(**o_data) if o_data else OutlierConfig(),
                    normalization=NormalizationConfig(**n_data) if n_data else NormalizationConfig(),
                    missing=MissingDataConfig(**m_data) if m_data else MissingDataConfig(),
                )
        return config

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "SensorDSPConfig":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
