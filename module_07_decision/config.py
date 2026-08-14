"""Configuration dataclasses for Module 7 — PhotonShield Decision / Task Output Layer."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any


@dataclass
class DecisionModelConfig:
    """Configuration for PhotonShield Multi-Task Model Heads.

    Attributes:
        d_model: Latent dimension size from Module 4 representation (pooled_output).
        enable_target: Whether Target Indication Head is active.
        enable_anomaly: Whether Anomaly Detection Head is active.
        enable_environment: Whether Environmental Assessment Head is active.
        num_target_classes: Number of classes for Target Indication (>= 2).
        target_hidden_dim: Optional hidden dimension for Target Head MLP.
        anomaly_hidden_dim: Optional hidden dimension for Anomaly Head MLP.
        environment_mode: 'regression' or 'classification'.
        num_environment_outputs: Number of continuous regression outputs (e.g. temp, humidity, pressure).
        num_environment_classes: Number of environment classes if mode is 'classification'.
        environment_hidden_dim: Optional hidden dimension for Environmental Head MLP.
        dropout: Dropout rate applied within task heads.
    """

    d_model: int = 128
    enable_target: bool = True
    enable_anomaly: bool = True
    enable_environment: bool = True
    num_target_classes: int = 2
    target_hidden_dim: Optional[int] = None
    anomaly_hidden_dim: Optional[int] = None
    environment_mode: Literal["regression", "classification"] = "regression"
    num_environment_outputs: int = 3
    num_environment_classes: int = 4
    environment_hidden_dim: Optional[int] = None
    dropout: float = 0.0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.enable_target and self.num_target_classes < 2:
            raise ValueError(f"num_target_classes must be >= 2, got {self.num_target_classes}")
        if self.enable_environment:
            if self.environment_mode == "regression" and self.num_environment_outputs <= 0:
                raise ValueError(f"num_environment_outputs must be > 0, got {self.num_environment_outputs}")
            if self.environment_mode == "classification" and self.num_environment_classes < 2:
                raise ValueError(f"num_environment_classes must be >= 2, got {self.num_environment_classes}")


@dataclass
class DecisionConfig:
    """Configuration for Application Decision Logic, Thresholds, and Smoothing.

    Attributes:
        target_threshold: Probability threshold for positive target decision (development default: 0.5).
        anomaly_threshold: Probability threshold for anomaly detection decision (development default: 0.5).
        smoothing_method: Causal decision smoothing filter ('none', 'ema', 'majority_vote').
        smoothing_window: Sliding window length for causal smoothing (timesteps).
        minimum_consecutive_detections: Required consecutive positive frames before triggering decision.
        hysteresis_enabled: Whether to apply dual-threshold hysteresis.
        hysteresis_on_threshold: Threshold to trigger positive event (ON state).
        hysteresis_off_threshold: Threshold to deactivate positive event (OFF state).
        calibration_method: Temperature scaling probability calibration ('none', 'temperature').
        temperature_target: Temperature parameter for Target Indication logits.
        temperature_anomaly: Temperature parameter for Anomaly Detection logits.
        event_combination_enabled: Whether to compute combined event state string.
        target_class_labels: Configurable target class label names dictionary.
        environment_output_names: Configurable environment output feature names list.
    """

    target_threshold: float = 0.5
    anomaly_threshold: float = 0.5
    smoothing_method: Literal["none", "ema", "majority_vote"] = "none"
    smoothing_window: int = 5
    minimum_consecutive_detections: int = 1
    hysteresis_enabled: bool = False
    hysteresis_on_threshold: float = 0.7
    hysteresis_off_threshold: float = 0.3
    calibration_method: Literal["none", "temperature"] = "none"
    temperature_target: float = 1.0
    temperature_anomaly: float = 1.0
    event_combination_enabled: bool = True
    target_class_labels: Dict[int, str] = field(default_factory=lambda: {0: "no_target", 1: "target"})
    environment_output_names: List[str] = field(
        default_factory=lambda: ["temperature", "humidity", "pressure"]
    )

    def validate(self) -> None:
        """Validate decision logic parameters."""
        if not (0.0 <= self.target_threshold <= 1.0):
            raise ValueError(f"target_threshold must be in [0, 1], got {self.target_threshold}")
        if not (0.0 <= self.anomaly_threshold <= 1.0):
            raise ValueError(f"anomaly_threshold must be in [0, 1], got {self.anomaly_threshold}")
        if self.smoothing_window < 1:
            raise ValueError(f"smoothing_window must be >= 1, got {self.smoothing_window}")
        if self.minimum_consecutive_detections < 1:
            raise ValueError(f"minimum_consecutive_detections must be >= 1, got {self.minimum_consecutive_detections}")
        if self.hysteresis_enabled:
            if not (self.hysteresis_off_threshold < self.hysteresis_on_threshold):
                raise ValueError(
                    f"hysteresis_off_threshold ({self.hysteresis_off_threshold}) must be < "
                    f"hysteresis_on_threshold ({self.hysteresis_on_threshold})"
                )
