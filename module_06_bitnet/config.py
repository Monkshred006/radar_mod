"""Configuration dataclasses for Module 6 — BitNet / 1.58-Bit Model Optimization."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Optional


@dataclass
class BitNetConfig:
    """Complete configuration for BitNet 1.58-bit ternary quantization and QAT.

    Defaults reflect the recommended initial research configuration:
      - Ternary weight matrices {-1, 0, +1}
      - Mean-absolute-weight per-tensor scaling
      - FP32 activation precision
      - Input projection, attention Q/K/V/O, and FFN layers ternarized
      - Mamba core, LayerNorm, and Task Head preserved at FP32
    """

    # ── Precision & Quantization Settings ────────────────────────────────────
    enabled: bool = True
    weight_bits: float = 1.58  # Conceptual label: log2(3) ≈ 1.585 bits
    ternary_values: List[int] = field(default_factory=lambda: [-1, 0, 1])

    # Scaling strategy
    scaling_method: Literal["mean_abs", "max_abs"] = "mean_abs"
    scaling_scope: Literal["per_tensor", "per_channel"] = "per_tensor"
    use_scale: bool = True

    # Activation precision (independent from weight precision)
    activation_precision: Literal["fp32", "fp16", "bf16", "int8"] = "fp32"

    # QAT settings
    train_master_weights: bool = True
    use_qat: bool = True

    # Layer Quantization Selection Flags
    quantize_input_projection: bool = True
    quantize_sensor_attention_qkv: bool = True
    quantize_sensor_attention_output: bool = True
    quantize_ffn: bool = True
    quantize_task_head: bool = False
    quantize_mamba_internal: bool = False
    quantize_layernorm: bool = False

    # Optional Ternary Packing
    packing_enabled: bool = False

    # Backend / Hardware
    backend: Literal["pytorch", "cuda", "edge_stub"] = "pytorch"

    # Initialization mode
    initialization_mode: Literal["fp32_converted", "fresh"] = "fp32_converted"

    # ── Training & Fine-Tuning Hyperparameters ────────────────────────────────
    epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    gradient_clip_norm: float = 1.0
    random_seed: int = 42

    # Directories
    checkpoint_dir: str = "checkpoints/bitnet"
    log_dir: str = "logs/bitnet"

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.scaling_method not in ("mean_abs", "max_abs"):
            raise ValueError(f"Invalid scaling_method: {self.scaling_method}")
        if self.scaling_scope not in ("per_tensor", "per_channel"):
            raise ValueError(f"Invalid scaling_scope: {self.scaling_scope}")
        if self.activation_precision not in ("fp32", "fp16", "bf16", "int8"):
            raise ValueError(f"Invalid activation_precision: {self.activation_precision}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
