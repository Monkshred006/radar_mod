"""Metadata structure and metadata utilities for Module 1 Radar Ingestion Pipeline."""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List


@dataclass
class RadarMetadata:
    """Dataclass holding metadata for a single radar frame or sequence.

    Unavailable metadata fields are explicitly represented as None.
    """

    radar_type: Optional[str] = None
    sampling_rate: Optional[float] = None  # Hz or Samples/sec
    num_antennas: Optional[int] = None
    frame_rate: Optional[float] = None  # FPS
    frame_dimensions: Optional[Tuple[int, ...]] = None
    timestamp: Optional[float] = None  # Seconds or Unix timestamp
    sequence_id: Optional[str] = None
    scene_id: Optional[str] = None
    dataset_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RadarMetadata":
        """Instantiate RadarMetadata from dictionary, replacing missing keys with None."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
