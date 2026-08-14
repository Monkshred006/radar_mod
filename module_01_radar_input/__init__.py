"""Module 1: Radar Data Input / Data Ingestion Package."""

from module_01_radar_input.config import RadarDatasetConfig
from module_01_radar_input.metadata import RadarMetadata
from module_01_radar_input.radar_loader import RadarLoader
from module_01_radar_input.dataset import RadarDataset, split_dataset, create_dataloaders
from module_01_radar_input.adapters.base import BaseRadarAdapter, DefaultDirectoryAdapter
from module_01_radar_input.validation import (
    validate_frame,
    validate_sequence,
    RadarDataValidationError,
    CorruptedFileError,
    InconsistentDimensionError
)

__all__ = [
    "RadarDatasetConfig",
    "RadarMetadata",
    "RadarLoader",
    "RadarDataset",
    "split_dataset",
    "create_dataloaders",
    "BaseRadarAdapter",
    "DefaultDirectoryAdapter",
    "validate_frame",
    "validate_sequence",
    "RadarDataValidationError",
    "CorruptedFileError",
    "InconsistentDimensionError",
]
