"""Base adapter interface and standard adapter implementation for radar datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
import numpy as np

from module_01_radar_input.metadata import RadarMetadata


class BaseRadarAdapter(ABC):
    """Abstract Base Class for Dataset Adapters.

    Adapters translate specific dataset directory layouts, custom metadata formats,
    or proprietary hardware structures into a unified index of frames/sequences.
    """

    @abstractmethod
    def discover_items(self, dataset_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Scan dataset path and return a list of discoverable frame/sequence descriptors.

        Each dictionary descriptor should contain at least:
            - 'frame_path': Path to raw frame file (or key inside archive)
            - 'scene_id': Identifier for the scene (for leakage-free splitting)
            - 'sequence_id': Optional sequence identifier
            - 'timestamp': Optional timestamp value
            - 'metadata': Optional custom metadata dictionary or RadarMetadata instance
        """
        pass

    @abstractmethod
    def parse_metadata(self, item_descriptor: Dict[str, Any]) -> RadarMetadata:
        """Extract or construct RadarMetadata from an item descriptor."""
        pass


class DefaultDirectoryAdapter(BaseRadarAdapter):
    """Default adapter for standard directory structures.

    Expected Directory Layout Options:
    1. Scene subdirectories: dataset_path/scene_01/frame_001.npy ...
    2. Flat directory: dataset_path/frame_001.npy (assigned default scene 'scene_default')
    """

    def __init__(self, supported_extensions: Tuple[str, ...] = (".npy", ".npz", ".csv", ".json")):
        self.supported_extensions = supported_extensions

    def discover_items(self, dataset_path: Union[str, Path]) -> List[Dict[str, Any]]:
        root = Path(dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")

        items = []
        # Check subdirectories as scenes first
        scene_dirs = [d for d in root.iterdir() if d.is_dir()]
        
        if scene_dirs:
            for s_dir in sorted(scene_dirs):
                scene_id = s_dir.name
                files = sorted([f for f in s_dir.iterdir() if f.suffix.lower() in self.supported_extensions])
                for idx, fpath in enumerate(files):
                    items.append({
                        "frame_path": fpath,
                        "scene_id": scene_id,
                        "sequence_id": f"{scene_id}_seq",
                        "frame_index": idx,
                        "timestamp": float(idx),
                        "metadata": {"dataset_source": "DefaultDirectoryAdapter"}
                    })
        else:
            # Flat directory layout
            files = sorted([f for f in root.iterdir() if f.suffix.lower() in self.supported_extensions])
            for idx, fpath in enumerate(files):
                items.append({
                    "frame_path": fpath,
                    "scene_id": "scene_default",
                    "sequence_id": "sequence_default",
                    "frame_index": idx,
                    "timestamp": float(idx),
                    "metadata": {"dataset_source": "DefaultDirectoryAdapter"}
                })

        return items

    def parse_metadata(self, item_descriptor: Dict[str, Any]) -> RadarMetadata:
        meta_dict = item_descriptor.get("metadata", {})
        return RadarMetadata(
            scene_id=item_descriptor.get("scene_id"),
            sequence_id=item_descriptor.get("sequence_id"),
            timestamp=item_descriptor.get("timestamp"),
            dataset_source=meta_dict.get("dataset_source", "DefaultDirectoryAdapter")
        )
