"""Configuration module for Module 1 Radar Ingestion Pipeline."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union, Dict, Any
import yaml


@dataclass
class RadarDatasetConfig:
    """Configuration settings for Radar DataLoader and Dataset.

    Attributes:
        dataset_path: Path to the dataset directory or manifest file.
        sequence_length: Number of frames per temporal sequence (T).
        frame_stride: Step size between frames within a single sequence.
            (e.g., frame_stride=1 takes consecutive frames [t, t+1, t+2];
                   frame_stride=2 takes [t, t+2, t+4]).
        sequence_stride: Step size between the start of consecutive sequences.
            (e.g., sequence_stride=1 produces [F0..F15], [F1..F16];
                   sequence_stride=16 produces non-overlapping sequences).
        train_ratio: Fraction of scenes/sequences used for training (0.0 to 1.0).
        val_ratio: Fraction of scenes/sequences used for validation (0.0 to 1.0).
        test_ratio: Fraction of scenes/sequences used for testing (0.0 to 1.0).
        batch_size: DataLoader batch size.
        num_workers: Number of subprocesses for PyTorch DataLoader.
        shuffle: Whether to shuffle sequence indices in DataLoader.
        random_seed: Seed for reproducible scene/sequence splitting and shuffling.
    """

    dataset_path: str = ""
    sequence_length: int = 16
    frame_stride: int = 1
    sequence_stride: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 4
    num_workers: int = 0
    shuffle: bool = True
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {self.sequence_length}")
        if self.frame_stride <= 0:
            raise ValueError(f"frame_stride must be > 0, got {self.frame_stride}")
        if self.sequence_stride <= 0:
            raise ValueError(f"sequence_stride must be > 0, got {self.sequence_stride}")

        total_ratio = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if not (0.999 <= total_ratio <= 1.001):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {self.train_ratio} + {self.val_ratio} + {self.test_ratio} = {total_ratio}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_yaml(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RadarDatasetConfig":
        """Instantiate config from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "RadarDatasetConfig":
        """Load configuration from YAML file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
