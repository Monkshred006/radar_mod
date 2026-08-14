"""Tests for config module."""

import pytest
from module_01_radar_input.config import RadarDatasetConfig


def test_config_defaults():
    config = RadarDatasetConfig(dataset_path="/tmp/radar")
    assert config.sequence_length == 16
    assert config.frame_stride == 1
    assert config.sequence_stride == 1
    assert config.train_ratio == 0.70


def test_invalid_config():
    with pytest.raises(ValueError, match="sequence_length must be > 0"):
        RadarDatasetConfig(sequence_length=0)

    with pytest.raises(ValueError, match="frame_stride must be > 0"):
        RadarDatasetConfig(frame_stride=-1)

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        RadarDatasetConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_config_yaml_io(tmp_path):
    yaml_file = tmp_path / "test_config.yaml"
    config = RadarDatasetConfig(dataset_path=str(tmp_path), sequence_length=8, batch_size=16)
    config.save_yaml(yaml_file)

    loaded = RadarDatasetConfig.from_yaml(yaml_file)
    assert loaded.sequence_length == 8
    assert loaded.batch_size == 16
    assert loaded.dataset_path == str(tmp_path)
