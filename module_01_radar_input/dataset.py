"""PyTorch Dataset implementation and scene-level splitting utilities."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from module_01_radar_input.config import RadarDatasetConfig
from module_01_radar_input.metadata import RadarMetadata
from module_01_radar_input.radar_loader import RadarLoader
from module_01_radar_input.adapters.base import BaseRadarAdapter, DefaultDirectoryAdapter
from module_01_radar_input.validation import validate_sequence, RadarDataValidationError


class RadarDataset(Dataset):
    """PyTorch Dataset for raw radar frame sequences.

    Exposes temporal sequences: [T, ...] tensor shape where T = sequence_length.
    Preserves original numerical dtype without automatic FP16 conversion.
    """

    def __init__(
        self,
        config: RadarDatasetConfig,
        adapter: Optional[BaseRadarAdapter] = None,
        loader: Optional[RadarLoader] = None,
        items: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Args:
            config: RadarDatasetConfig instance containing paths and sequence params.
            adapter: Dataset format adapter. Defaults to DefaultDirectoryAdapter.
            loader: Radar frame loader. Defaults to RadarLoader().
            items: Explicit list of discovered frame items (used when constructing sub-splits).
        """
        self.config = config
        self.adapter = adapter or DefaultDirectoryAdapter()
        self.loader = loader or RadarLoader()

        if items is not None:
            self.discovered_items = items
        else:
            self.discovered_items = self.adapter.discover_items(self.config.dataset_path)

        # Build temporal sequence index map
        self.sequences = self._build_sequence_indices()

    def _build_sequence_indices(self) -> List[List[Dict[str, Any]]]:
        """Construct sequence index windows per scene to avoid temporal sequence mixing across scenes.

        Sequence configuration parameters:
        - sequence_length: number of frames per sequence (T)
        - frame_stride: step size between consecutive frames in sequence
        - sequence_stride: step size between start of consecutive sequences
        """
        # Group items by scene_id
        scenes: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.discovered_items:
            scene_id = item.get("scene_id", "default_scene")
            scenes.setdefault(scene_id, []).append(item)

        sequence_list = []
        seq_len = self.config.sequence_length
        f_stride = self.config.frame_stride
        s_stride = self.config.sequence_stride

        # Minimum required frames span for a valid sequence
        required_span = (seq_len - 1) * f_stride + 1

        for scene_id, frame_items in scenes.items():
            n_frames = len(frame_items)
            if n_frames < required_span:
                continue

            start_idx = 0
            while start_idx + required_span <= n_frames:
                # Take sequence frames according to frame_stride
                seq_frames = [
                    frame_items[start_idx + i * f_stride]
                    for i in range(seq_len)
                ]
                sequence_list.append(seq_frames)
                start_idx += s_stride

        return sequence_list

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve structured sample containing:
        {
            "radar": torch.Tensor,       # Shape [T, ...] preserving original numerical dtype/representation
            "timestamp": torch.Tensor,   # Shape [T]
            "metadata": Dict[str, Any]   # Aggregated metadata for sequence
        }
        """
        seq_items = self.sequences[idx]

        frames = []
        timestamps = []
        metadata_list = []

        for item in seq_items:
            frame_arr = self.loader.load_frame(item["frame_path"])
            frames.append(frame_arr)
            timestamps.append(item.get("timestamp", 0.0))
            meta = self.adapter.parse_metadata(item)
            # Filter None fields so PyTorch default_collate won't crash on NoneType
            meta_dict = {k: (v if v is not None else "") for k, v in meta.to_dict().items()}
            metadata_list.append(meta_dict)

        # Validate sequence dimension consistency
        validate_sequence(frames)

        # Stack into numpy array shape [T, ...]
        stacked_np = np.stack(frames, axis=0)

        # Convert safely to PyTorch tensor while preserving original dtype (e.g. float32, float64, int, complex)
        radar_tensor = torch.from_numpy(stacked_np)

        return {
            "radar": radar_tensor,
            "timestamp": torch.tensor(timestamps, dtype=torch.float64),
            "metadata": {
                "sequence_id": seq_items[0].get("sequence_id", ""),
                "scene_id": seq_items[0].get("scene_id", ""),
                "frame_metadata": metadata_list
            }
        }


def split_dataset(
    dataset: RadarDataset,
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[RadarDataset, RadarDataset, RadarDataset]:
    """Split dataset into Train, Validation, and Test sets based on SCENE IDs.

    Guarantees strict scene-level partition so all frames/sequences belonging
    to a scene reside strictly within one split to prevent leakage.

    Saves split manifest (split_info.json) if output_dir is provided.
    """
    config = dataset.config
    discovered = dataset.discovered_items

    # Group discovered items by scene_id
    scene_map: Dict[str, List[Dict[str, Any]]] = {}
    for item in discovered:
        s_id = item.get("scene_id", "default_scene")
        scene_map.setdefault(s_id, []).append(item)

    unique_scenes = sorted(list(scene_map.keys()))

    # Deterministic shuffle of scene IDs using config.random_seed
    rng = random.Random(config.random_seed)
    shuffled_scenes = list(unique_scenes)
    rng.shuffle(shuffled_scenes)

    n_scenes = len(shuffled_scenes)
    n_train = int(round(n_scenes * config.train_ratio))
    n_val = int(round(n_scenes * config.val_ratio))

    train_scenes = set(shuffled_scenes[:n_train])
    val_scenes = set(shuffled_scenes[n_train:n_train + n_val])
    test_scenes = set(shuffled_scenes[n_train + n_val:])

    train_items = [item for item in discovered if item.get("scene_id") in train_scenes]
    val_items = [item for item in discovered if item.get("scene_id") in val_scenes]
    test_items = [item for item in discovered if item.get("scene_id") in test_scenes]

    train_ds = RadarDataset(config, adapter=dataset.adapter, loader=dataset.loader, items=train_items)
    val_ds = RadarDataset(config, adapter=dataset.adapter, loader=dataset.loader, items=val_items)
    test_ds = RadarDataset(config, adapter=dataset.adapter, loader=dataset.loader, items=test_items)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        split_manifest = {
            "random_seed": config.random_seed,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "scene_assignments": {
                "train": list(train_scenes),
                "val": list(val_scenes),
                "test": list(test_scenes)
            },
            "sequence_counts": {
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds)
            }
        }
        with open(out_path / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2)

    return train_ds, val_ds, test_ds


def create_dataloaders(
    train_ds: RadarDataset,
    val_ds: RadarDataset,
    test_ds: RadarDataset
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, val, and test datasets."""
    config = train_ds.config

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, val_loader, test_loader
