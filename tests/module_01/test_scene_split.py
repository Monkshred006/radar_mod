"""Dedicated test suite verifying scene-level dataset splitting to eliminate temporal/spatial leakage."""

import json
import pytest
import numpy as np
from module_01_radar_input.config import RadarDatasetConfig
from module_01_radar_input.dataset import RadarDataset, split_dataset


@pytest.fixture
def multi_scene_dataset(tmp_path):
    """Generate 10 scenes, each containing 10 frames."""
    for s_idx in range(10):
        scene_dir = tmp_path / f"scene_{s_idx:02d}"
        scene_dir.mkdir()
        for f_idx in range(10):
            frame = np.ones((8, 8), dtype=np.float32) * (s_idx * 10 + f_idx)
            np.save(scene_dir / f"frame_{f_idx:02d}.npy", frame)
    return tmp_path


def test_scene_level_split_no_leakage(multi_scene_dataset, tmp_path):
    config = RadarDatasetConfig(
        dataset_path=str(multi_scene_dataset),
        sequence_length=4,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_seed=42
    )

    dataset = RadarDataset(config)
    train_ds, val_ds, test_ds = split_dataset(dataset, output_dir=tmp_path / "splits")

    train_scenes = set(it["scene_id"] for it in train_ds.discovered_items)
    val_scenes = set(it["scene_id"] for it in val_ds.discovered_items)
    test_scenes = set(it["scene_id"] for it in test_ds.discovered_items)

    # 1. Assert disjoint scene sets (ZERO LEAKAGE)
    assert train_scenes.isdisjoint(val_scenes), f"Leakage detected between train & val: {train_scenes & val_scenes}"
    assert train_scenes.isdisjoint(test_scenes), f"Leakage detected between train & test: {train_scenes & test_scenes}"
    assert val_scenes.isdisjoint(test_scenes), f"Leakage detected between val & test: {val_scenes & test_scenes}"

    # 2. Total scenes should sum to 10
    total_scenes = len(train_scenes) + len(val_scenes) + len(test_scenes)
    assert total_scenes == 10
    assert len(train_scenes) == 6
    assert len(val_scenes) == 2
    assert len(test_scenes) == 2

    # 3. Check split_info.json artifact creation
    manifest_path = tmp_path / "splits" / "split_info.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["random_seed"] == 42
    assert set(manifest["scene_assignments"]["train"]) == train_scenes
    assert set(manifest["scene_assignments"]["val"]) == val_scenes
    assert set(manifest["scene_assignments"]["test"]) == test_scenes
