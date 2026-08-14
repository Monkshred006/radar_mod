"""Dataset inspection CLI tool.

Run via:
python -m module_01_radar_input.inspect --dataset-path <path>
"""

import argparse
from pathlib import Path
from module_01_radar_input.config import RadarDatasetConfig
from module_01_radar_input.dataset import RadarDataset, split_dataset
from module_01_radar_input.validation import validate_file_exists


def inspect_dataset(dataset_path: str, config_yaml: str = None) -> None:
    """Inspect dataset and display statistics."""
    path = Path(dataset_path)
    print("=" * 60)
    print("      MODULE 1: RADAR DATASET INSPECTION REPORT      ")
    print("=" * 60)

    if config_yaml and Path(config_yaml).exists():
        config = RadarDatasetConfig.from_yaml(config_yaml)
        config.dataset_path = str(path)
    else:
        config = RadarDatasetConfig(dataset_path=str(path))

    dataset = RadarDataset(config)
    
    total_discovered_files = len(dataset.discovered_items)
    total_sequences = len(dataset)

    # Gather unique scenes
    scenes = set(item.get("scene_id", "default") for item in dataset.discovered_items)

    print(f"Dataset Path         : {path.resolve()}")
    print(f"Total Discovered Files: {total_discovered_files}")
    print(f"Total Scenes         : {len(scenes)}")
    print(f"Total Sequences      : {total_sequences}")
    print(f"Sequence Length (T)  : {config.sequence_length}")
    print(f"Frame Stride         : {config.frame_stride}")
    print(f"Sequence Stride      : {config.sequence_stride}")

    if total_sequences > 0:
        sample = dataset[0]
        radar_tensor = sample["radar"]
        print(f"Sample Radar Tensor Shape: {list(radar_tensor.shape)} (Format: [T, ...])")
        print(f"Sample Radar Tensor Dtype: {radar_tensor.dtype}")

        # Split stats
        train_ds, val_ds, test_ds = split_dataset(dataset)
        print("-" * 60)
        print("Scene-Level Dataset Split Summary:")
        print(f"  Train Sequences : {len(train_ds)}")
        print(f"  Val Sequences   : {len(val_ds)}")
        print(f"  Test Sequences  : {len(test_ds)}")
    else:
        print("WARNING: No valid sequences could be formed from the dataset path.")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Radar Dataset (Module 1)")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to radar dataset directory")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config YAML")

    args = parser.parse_args()
    inspect_dataset(args.dataset_path, args.config)
