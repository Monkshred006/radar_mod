"""Configuration dataclasses for Module 5 — FP32 Training + Evaluation Pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, List


@dataclass
class TrainingConfig:
    """Complete training configuration for PhotonShield Mamba-Hybrid FP32 training.

    Defaults are conservative and correct for a first FP32 baseline run.
    """

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset_path: str = ""
    channel_names: List[str] = field(default_factory=lambda: [
        "photodiode_1", "photodiode_2",
        "temperature", "humidity", "pressure",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "distance",
    ])
    sequence_length: int = 20
    frame_stride: int = 1
    sequence_stride: int = 1
    precompute_features: bool = False   # If True, cache Module 3 features to disk
    cache_dir: str = ""                 # Where to write/read precomputed features

    # ── Optimization ─────────────────────────────────────────────────────────
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    sgd_momentum: float = 0.9

    # ── Scheduler ────────────────────────────────────────────────────────────
    scheduler: Literal["cosine", "step", "plateau", "none"] = "cosine"
    scheduler_step_size: int = 10       # for StepLR
    scheduler_gamma: float = 0.5        # for StepLR
    scheduler_min_lr: float = 1e-6     # for CosineAnnealingLR

    # ── Gradient management ───────────────────────────────────────────────────
    gradient_clip_norm: float = 1.0     # 0.0 = disabled
    gradient_accumulation_steps: int = 1

    # ── Device ───────────────────────────────────────────────────────────────
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = 0

    # ── Precision (MUST remain FP32 for the reference baseline) ──────────────
    # mixed_precision = False is the required default.
    # Do NOT set to True unless you intend an experimental run.
    mixed_precision: bool = False

    # ── Target ───────────────────────────────────────────────────────────────
    target_type: Literal["regression", "classification", "multitask"] = "regression"
    num_classes: int = 2
    num_regression_outputs: int = 1

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_name: str = "mse"
    loss_kwargs: dict = field(default_factory=dict)

    # ── Model output selection ────────────────────────────────────────────────
    output_key: str = "pooled_output"   # which Module 4 output tensor to use

    # ── Reproducibility ───────────────────────────────────────────────────────
    random_seed: int = 42

    # ── Checkpointing ────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    save_latest: bool = True
    resume_from: str = ""               # Path to checkpoint for resume

    # ── Logging ──────────────────────────────────────────────────────────────
    log_dir: str = "logs"

    # ── Validation & early stopping ───────────────────────────────────────────
    val_every_n_epochs: int = 1
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: Literal["min", "max"] = "min"
    early_stopping_min_delta: float = 1e-5

    def validate(self) -> None:
        """Basic configuration validation."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.gradient_clip_norm < 0:
            raise ValueError(f"gradient_clip_norm must be >= 0")
        if self.mixed_precision:
            import warnings
            warnings.warn(
                "mixed_precision=True is not the FP32 reference baseline. "
                "Ensure this is intentional.",
                UserWarning,
                stacklevel=2,
            )
