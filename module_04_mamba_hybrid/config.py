"""Configuration dataclasses for Module 4 — PhotonShield Mamba-Hybrid Engine."""

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class TaskHeadConfig:
    """Configuration for task heads."""
    head_type: Literal["classification", "regression", "multitask"] = "classification"
    num_classes: int = 2
    num_regression_outputs: int = 1
    dropout: float = 0.0
    hidden_dim: Optional[int] = None


@dataclass
class MambaHybridConfig:
    """Configuration for PhotonShield Mamba-Hybrid Engine.

    Supports configurable dimensions, Mamba SSM parameters, sensor interaction
    attention heads, pooling methods, and ablation switches.
    """
    # Dimension settings
    d_model: int = 128
    sensor_feature_dim: int = 48  # D_features from Module 3 tokens [T, S, D_features]
    num_sensor_groups: int = 5    # S groups from Module 3 (optical, env, motion, dist, quality)
    fused_feature_dim: int = 101  # F_fused from Module 3 flat features [T, F_fused]

    # Model architecture settings
    num_layers: int = 2
    ffn_multiplier: int = 2
    dropout: float = 0.0
    normalization: str = "layernorm"

    # Mamba SSM settings
    mamba_state_dim: int = 16     # State dimension d_state
    mamba_conv_dim: int = 4       # d_conv 1D convolution kernel size
    mamba_expand: int = 2         # Expansion factor in Mamba block
    backend: Literal["auto", "mamba-ssm", "fallback"] = "auto"

    # Sensor Interaction Branch settings
    num_attention_heads: int = 2
    attention_dropout: float = 0.0

    # Temporal / Positional Encoding settings
    use_temporal_encoding: bool = True
    temporal_encoding_type: Literal["learned", "timestamp_delta", "none"] = "learned"
    max_sequence_length: int = 1024

    # Pooling settings
    pooling: Literal["masked_mean", "mean", "last"] = "masked_mean"

    # Ablation switches
    use_mamba: bool = True
    use_sensor_attention: bool = True

    # Default task head config
    head_config: TaskHeadConfig = field(default_factory=TaskHeadConfig)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.use_sensor_attention and self.d_model % self.num_attention_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_attention_heads "
                f"({self.num_attention_heads})"
            )
        if self.pooling not in ("masked_mean", "mean", "last"):
            raise ValueError(f"Invalid pooling method: {self.pooling}")
        if self.backend not in ("auto", "mamba-ssm", "fallback"):
            raise ValueError(f"Invalid backend: {self.backend}")
