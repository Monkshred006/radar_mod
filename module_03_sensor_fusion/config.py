"""Configuration dataclasses for Module 3 — Sensor Fusion + Feature Extraction."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
import yaml


@dataclass
class OpticalFeatureConfig:
    """Config for photodiode/optical feature extraction."""
    enabled: bool = True
    amplitude: bool = True
    first_diff: bool = True
    rate_of_change: bool = True
    rolling_mean: bool = True
    rolling_std: bool = True
    rolling_energy: bool = True
    window_size: int = 5


@dataclass
class EnvironmentalFeatureConfig:
    """Config for BME280 environmental feature extraction."""
    enabled: bool = True
    current_val: bool = True
    first_diff: bool = True
    rolling_mean: bool = True
    window_size: int = 5


@dataclass
class MotionFeatureConfig:
    """Config for MPU6050 motion feature extraction."""
    enabled: bool = True
    raw_channels: bool = True
    accel_magnitude: bool = True
    gyro_magnitude: bool = True
    first_diff: bool = True
    rolling_mean: bool = True
    rolling_std: bool = True
    window_size: int = 5


@dataclass
class DistanceFeatureConfig:
    """Config for VL53L0X distance feature extraction."""
    enabled: bool = True
    current_val: bool = True
    first_diff: bool = True
    rate_of_change: bool = True
    rolling_mean: bool = True
    window_size: int = 5


@dataclass
class QualityFeatureConfig:
    """Config for signal quality feature encoding."""
    enabled: bool = True
    include_outlier_mask: bool = True
    include_missing_mask: bool = True
    include_interpolated_mask: bool = True
    include_valid_flag: bool = True


@dataclass
class TokenizerConfig:
    """Config for Sensor-Aware Tokenization layout.

    Token layout produced: [B, T, S, D_features]
    S: number of active sensor groups (e.g. 5: optical, env, motion, dist, quality)
    D_features: feature vector dimension per sensor group (padded to max if padding=True)
    """
    pad_to_max_dim: bool = True
    explicit_group_order: List[str] = field(
        default_factory=lambda: ["optical", "environment", "motion", "distance", "quality"]
    )


@dataclass
class Module3Config:
    """Top-level configuration for Module 3 Sensor Fusion pipeline.

    Attributes:
        optical: OpticalFeatureConfig
        environmental: EnvironmentalFeatureConfig
        motion: MotionFeatureConfig
        distance: DistanceFeatureConfig
        quality: QualityFeatureConfig
        tokenizer: TokenizerConfig
        streaming: Enforce strict causality in streaming mode.
        dtype: Output torch dtype ("float32" or "float64").
    """
    optical: OpticalFeatureConfig = field(default_factory=OpticalFeatureConfig)
    environmental: EnvironmentalFeatureConfig = field(default_factory=EnvironmentalFeatureConfig)
    motion: MotionFeatureConfig = field(default_factory=MotionFeatureConfig)
    distance: DistanceFeatureConfig = field(default_factory=DistanceFeatureConfig)
    quality: QualityFeatureConfig = field(default_factory=QualityFeatureConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    streaming: bool = False
    dtype: Literal["float32", "float64"] = "float32"

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_yaml(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict) -> "Module3Config":
        opt_data = data.pop("optical", {})
        env_data = data.pop("environmental", {})
        mot_data = data.pop("motion", {})
        dst_data = data.pop("distance", {})
        q_data = data.pop("quality", {})
        tok_data = data.pop("tokenizer", {})

        return cls(
            optical=OpticalFeatureConfig(**opt_data) if opt_data else OpticalFeatureConfig(),
            environmental=EnvironmentalFeatureConfig(**env_data) if env_data else EnvironmentalFeatureConfig(),
            motion=MotionFeatureConfig(**mot_data) if mot_data else MotionFeatureConfig(),
            distance=DistanceFeatureConfig(**dst_data) if dst_data else DistanceFeatureConfig(),
            quality=QualityFeatureConfig(**q_data) if q_data else QualityFeatureConfig(),
            tokenizer=TokenizerConfig(**tok_data) if tok_data else TokenizerConfig(),
            **data,
        )

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "Module3Config":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
