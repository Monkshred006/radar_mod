"""Unit tests for RaDICaL dataset adapter and feature extraction."""

import numpy as np
import pytest
import torch

from module_03_sensor_fusion.radical_adapter import (
    RaDICaLDatasetAdapter,
    RaDICaLDataset,
    RaDICaLFeatureExtractor,
)


class TestRaDICaLAdapter:
    """Test suite for RaDICaL adapter."""

    def test_feature_extractor_shape(self):
        extractor = RaDICaLFeatureExtractor(feature_dim=64, normalization="zscore")
        dummy_rd = np.random.rand(64, 32)
        feat = extractor.extract(dummy_rd)
        assert feat.shape == (64,)
        assert not np.isnan(feat).any()

    def test_feature_extractor_normalization_modes(self):
        dummy_rd = np.random.rand(64, 32) * 100.0
        for mode in ["zscore", "minmax", "db"]:
            extractor = RaDICaLFeatureExtractor(feature_dim=64, normalization=mode)
            feat = extractor.extract(dummy_rd)
            assert feat.shape == (64,)
            assert not np.isnan(feat).any()

    def test_synthetic_sample_generation(self):
        adapter = RaDICaLDatasetAdapter(
            sequence_length=16,
            feature_dim=64,
            num_classes=4,
            seed=42,
        )
        feats, det, cls_lbl, ano = adapter.generate_synthetic_samples(num_samples=20)
        assert feats.shape == (20, 16, 64)
        assert det.shape == (20, 1)
        assert cls_lbl.shape == (20,)
        assert ano.shape == (20, 1)

    def test_dataset_splits_and_dataloaders(self):
        adapter = RaDICaLDatasetAdapter(
            sequence_length=16,
            feature_dim=64,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        train_ds, val_ds, test_ds = adapter.get_datasets(num_synthetic_fallback=100)
        assert len(train_ds) == 70
        assert len(val_ds) == 15
        assert len(test_ds) == 15

        train_loader, val_loader, test_loader = adapter.get_dataloaders(
            batch_size=16, num_synthetic_fallback=100
        )
        batch = next(iter(train_loader))
        assert batch["features"].shape == (16, 16, 64)
        assert batch["detection"].shape == (16, 1)
        assert batch["classification"].shape == (16,)
        assert batch["anomaly"].shape == (16, 1)
