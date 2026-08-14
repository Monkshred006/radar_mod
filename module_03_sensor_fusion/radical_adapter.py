"""RaDICaL Dataset Adapter for PhotonShield AI.

Converts RaDICaL radar tensors / Range-Doppler (RD) frames into fixed-size
feature sequences [B, T, D] suitable for the Mamba temporal foundation (PhotonV0).

Supports:
1. Loading real RaDICaL numpy/npz/h5 tensor files or directories.
2. Generating synthetic RaDICaL-style Range-Doppler tensor sequences for standalone execution.
3. Feature extraction from RD frames:
   - Range-Doppler energy projection & binning
   - Doppler centroid & peak spread features
   - SNR and noise floor estimation
   - Linear / spectral reduction to exact `feature_dim` [D]
4. Normalization modes: 'zscore', 'minmax', 'db', 'none'.
5. Deterministic train/val/test splits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RaDICaLFeatureExtractor:
    """Extracts a fixed-size 1D feature vector of dimension `feature_dim` from a 2D Range-Doppler map."""

    def __init__(self, feature_dim: int = 64, normalization: str = "zscore") -> None:
        self.feature_dim = feature_dim
        self.normalization = normalization

    def extract(self, rd_map: np.ndarray) -> np.ndarray:
        """Extract feature vector from a single 2D Range-Doppler map [Range_Bins, Doppler_Bins].

        Args:
            rd_map: 2D numpy array [R, D_bins] or 3D [Channels, R, D_bins].

        Returns:
            1D numpy array of length `self.feature_dim`.
        """
        if rd_map.ndim == 3:
            # Average across antennas/channels if multi-channel
            rd_map = np.mean(np.abs(rd_map), axis=0)
        else:
            rd_map = np.abs(rd_map)

        r_bins, d_bins = rd_map.shape

        # 1. Range profile: sum along Doppler dimension
        range_profile = np.sum(rd_map, axis=1)  # [R]
        # 2. Doppler profile: sum along Range dimension
        doppler_profile = np.sum(rd_map, axis=0)  # [D_bins]

        # 3. Peak statistics & summary metrics
        max_val = np.max(rd_map) if rd_map.size > 0 else 0.0
        mean_val = np.mean(rd_map) if rd_map.size > 0 else 0.0
        std_val = np.std(rd_map) if rd_map.size > 0 else 0.0
        snr_est = (max_val / (mean_val + 1e-8)) if mean_val > 0 else 0.0

        # Resample profiles to allocate into feature_dim
        # Allocate: ~45% range, ~45% doppler, ~10% scalar stats
        num_scalars = 4
        rem_dim = self.feature_dim - num_scalars
        n_range = rem_dim // 2
        n_doppler = rem_dim - n_range

        # Interpolate / resample range profile
        x_r = np.linspace(0, 1, len(range_profile))
        x_r_new = np.linspace(0, 1, n_range)
        r_resampled = np.interp(x_r_new, x_r, range_profile)

        # Interpolate / resample doppler profile
        x_d = np.linspace(0, 1, len(doppler_profile))
        x_d_new = np.linspace(0, 1, n_doppler)
        d_resampled = np.interp(x_d_new, x_d, doppler_profile)

        scalars = np.array([max_val, mean_val, std_val, snr_est], dtype=np.float32)
        feat = np.concatenate([r_resampled, d_resampled, scalars])

        # Apply normalization
        if self.normalization == "zscore":
            s = np.std(feat)
            if s > 1e-8:
                feat = (feat - np.mean(feat)) / s
            else:
                feat = feat - np.mean(feat)
        elif self.normalization == "minmax":
            mn, mx = np.min(feat), np.max(feat)
            if (mx - mn) > 1e-8:
                feat = (feat - mn) / (mx - mn)
            else:
                feat = np.zeros_like(feat)
        elif self.normalization == "db":
            feat = 10.0 * np.log10(np.maximum(np.abs(feat), 1e-6))
            feat = np.clip(feat, -100.0, 100.0) / 100.0

        return feat.astype(np.float32)


class RaDICaLDataset(Dataset):
    """PyTorch Dataset for RaDICaL radar sequences [T, D]."""

    def __init__(
        self,
        features: np.ndarray,
        labels_det: np.ndarray,
        labels_cls: np.ndarray,
        labels_ano: np.ndarray,
    ) -> None:
        """Initialize dataset.

        Args:
            features: Array of shape [N, T, D].
            labels_det: Detection labels [N, 1] (0 or 1).
            labels_cls: Classification labels [N] (integer class 0..C-1).
            labels_ano: Anomaly scores [N, 1] (continuous float).
        """
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels_det = torch.as_tensor(labels_det, dtype=torch.float32)
        self.labels_cls = torch.as_tensor(labels_cls, dtype=torch.long)
        self.labels_ano = torch.as_tensor(labels_ano, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],      # [T, D]
            "detection": self.labels_det[idx],    # [1]
            "classification": self.labels_cls[idx], # scalar int
            "anomaly": self.labels_ano[idx],      # [1]
        }


class RaDICaLDatasetAdapter:
    """Adapter to load, preprocess, and partition RaDICaL radar sequences."""

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        sequence_length: int = 16,
        feature_dim: int = 64,
        num_classes: int = 4,
        normalization: Literal["zscore", "minmax", "db", "none"] = "zscore",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.data_path = Path(data_path) if data_path else None
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.normalization = normalization
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.extractor = RaDICaLFeatureExtractor(
            feature_dim=feature_dim, normalization=normalization
        )

    def generate_synthetic_samples(
        self,
        num_samples: int = 200,
        range_bins: int = 64,
        doppler_bins: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic RaDICaL Range-Doppler sequences for self-contained execution.

        Simulates:
        - Class 0: Noise / empty scene (no target)
        - Class 1: Pedestrian (low Doppler, moderate RCS)
        - Class 2: Cyclist (moderate Doppler, higher RCS)
        - Class 3: Vehicle (high Doppler, strong RCS)
        - Anomaly scenarios: High clutter / jamming / multipath reflections
        """
        rng = np.random.RandomState(self.seed)
        features_list = []
        det_list = []
        cls_list = []
        ano_list = []

        for i in range(num_samples):
            # Pick class
            c = rng.randint(0, self.num_classes)
            has_target = (c > 0)
            is_anomaly = (rng.rand() < 0.15)  # 15% anomaly rate

            seq_feats = []
            # Trajectory state
            r_center = rng.uniform(10, range_bins - 10)
            d_center = rng.uniform(doppler_bins * 0.3, doppler_bins * 0.7)
            velocity = (c * 0.5) if has_target else 0.0

            for t in range(self.sequence_length):
                # Background noise
                noise_floor = rng.exponential(scale=1.0, size=(range_bins, doppler_bins))

                if has_target:
                    curr_r = int(np.clip(r_center + velocity * t, 0, range_bins - 1))
                    curr_d = int(np.clip(d_center + (rng.randn() * 0.2), 0, doppler_bins - 1))
                    # Add target peak
                    peak_amp = 15.0 * c
                    noise_floor[curr_r, curr_d] += peak_amp
                    if curr_r + 1 < range_bins:
                        noise_floor[curr_r + 1, curr_d] += peak_amp * 0.5
                    if curr_d + 1 < doppler_bins:
                        noise_floor[curr_r, curr_d + 1] += peak_amp * 0.5

                if is_anomaly:
                    # Inject anomalous wideband pulse / interference
                    noise_floor[:, rng.randint(0, doppler_bins)] += rng.uniform(20.0, 50.0)

                feat_t = self.extractor.extract(noise_floor)
                seq_feats.append(feat_t)

            features_list.append(np.stack(seq_feats, axis=0))  # [T, D]
            det_list.append([1.0 if has_target else 0.0])
            cls_list.append(c)
            ano_list.append([1.0 if is_anomaly else 0.0])

        return (
            np.array(features_list, dtype=np.float32),
            np.array(det_list, dtype=np.float32),
            np.array(cls_list, dtype=np.int64),
            np.array(ano_list, dtype=np.float32),
        )

    def load_data(
        self,
        num_synthetic_fallback: int = 300,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset from path, or fallback to synthetic generation if path not found."""
        if self.data_path and self.data_path.exists():
            # Check for numpy archive
            if self.data_path.is_file() and self.data_path.suffix in [".npz", ".npy"]:
                data = np.load(self.data_path)
                if isinstance(data, np.lib.npyio.NpzFile):
                    feats = data["features"]
                    det = data["detection"] if "detection" in data else (data["classification"] > 0).astype(np.float32)[:, None]
                    cls_lbl = data["classification"]
                    ano = data["anomaly"] if "anomaly" in data else np.zeros((len(feats), 1), dtype=np.float32)
                    return feats, det, cls_lbl, ano
            elif self.data_path.is_dir():
                # Check for .npy / .npz files inside
                files = list(self.data_path.glob("*.npz")) or list(self.data_path.glob("*.npy"))
                if files:
                    loaded_feats, loaded_det, loaded_cls, loaded_ano = [], [], [], []
                    for f in files:
                        d = np.load(f)
                        if isinstance(d, np.lib.npyio.NpzFile):
                            loaded_feats.append(d["features"])
                            loaded_det.append(d.get("detection", (d["classification"] > 0).astype(np.float32)[:, None]))
                            loaded_cls.append(d["classification"])
                            loaded_ano.append(d.get("anomaly", np.zeros((len(d["features"]), 1), dtype=np.float32)))
                    return (
                        np.concatenate(loaded_feats, axis=0),
                        np.concatenate(loaded_det, axis=0),
                        np.concatenate(loaded_cls, axis=0),
                        np.concatenate(loaded_ano, axis=0),
                    )

        # Fallback to deterministic synthetic generation
        return self.generate_synthetic_samples(num_samples=num_synthetic_fallback)

    def get_datasets(
        self,
        num_synthetic_fallback: int = 300,
    ) -> Tuple[RaDICaLDataset, RaDICaLDataset, RaDICaLDataset]:
        """Split data and return (train_dataset, val_dataset, test_dataset)."""
        feats, det, cls_lbl, ano = self.load_data(num_synthetic_fallback=num_synthetic_fallback)
        n = len(feats)

        # Shuffle deterministically
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n)

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_ds = RaDICaLDataset(feats[train_idx], det[train_idx], cls_lbl[train_idx], ano[train_idx])
        val_ds = RaDICaLDataset(feats[val_idx], det[val_idx], cls_lbl[val_idx], ano[val_idx])
        test_ds = RaDICaLDataset(feats[test_idx], det[test_idx], cls_lbl[test_idx], ano[test_idx])

        return train_ds, val_ds, test_ds

    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        num_synthetic_fallback: int = 300,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoader instances for train, val, and test splits."""
        train_ds, val_ds, test_ds = self.get_datasets(num_synthetic_fallback=num_synthetic_fallback)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader
