"""Tests for RadarDataset and PyTorch DataLoader integration."""

import pytest
import numpy as np
import torch
from module_01_radar_input.config import RadarDatasetConfig
from module_01_radar_input.dataset import RadarDataset, create_dataloaders, split_dataset


@pytest.fixture
def synthetic_dataset_dir(tmp_path):
    scene = tmp_path / "scene_01"
    scene.mkdir()
    # Save 20 frames
    for i in range(20):
        arr = (np.ones((16, 8), dtype=np.float32) * i)
        np.save(scene / f"frame_{i:03d}.npy", arr)
    return tmp_path


def test_dataset_sequence_indexing(synthetic_dataset_dir):
    config = RadarDatasetConfig(
        dataset_path=str(synthetic_dataset_dir),
        sequence_length=5,
        frame_stride=1,
        sequence_stride=1
    )
    dataset = RadarDataset(config)

    # 20 frames with seq_len=5 and stride=1 -> 16 sequences
    assert len(dataset) == 16

    sample = dataset[0]
    radar_tensor = sample["radar"]
    assert isinstance(radar_tensor, torch.Tensor)
    assert radar_tensor.shape == (5, 16, 8)
    assert radar_tensor.dtype == torch.float32

    # Check frame stride = 2
    config_stride2 = RadarDatasetConfig(
        dataset_path=str(synthetic_dataset_dir),
        sequence_length=5,
        frame_stride=2,
        sequence_stride=1
    )
    dataset_stride2 = RadarDataset(config_stride2)
    # 5 frames spanning (5-1)*2+1 = 9 frames. 20 - 9 + 1 = 12 sequences
    assert len(dataset_stride2) == 12


def test_pytorch_dataloader_batching(synthetic_dataset_dir):
    config = RadarDatasetConfig(
        dataset_path=str(synthetic_dataset_dir),
        sequence_length=4,
        batch_size=2
    )
    dataset = RadarDataset(config)
    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_loader, _, _ = create_dataloaders(train_ds, val_ds, test_ds)

    for batch in train_loader:
        radar_batch = batch["radar"]
        assert radar_batch.ndim == 4  # [B, T, H, W] -> [2, 4, 16, 8]
        assert radar_batch.shape[0] <= 2
        assert radar_batch.shape[1] == 4
        break
