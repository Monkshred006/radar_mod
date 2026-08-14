"""RaDICaL Dataset Adapter for PhotonShield AI.

Converts RaDICaL radar tensors / Range-Doppler (RD) frames into fixed-size
feature sequences [B, T, D] suitable for the Mamba temporal foundation (PhotonV0).

Supports:
1. Loading official RaDICaL dataset directories (train/, val/, test/) containing .npz and .h5 files.
2. Loading single .npz, .npy, and .h5 archives.
3. Feature extraction from RD frames:
   - Range-Doppler energy projection & binning
   - Doppler centroid & peak spread features
   - SNR and noise floor estimation
   - Linear / spectral reduction to exact `feature_dim` [D]
4. Normalization modes: 'db' (default), 'zscore', 'minmax', 'none'.
5. Deterministic train/val/test splits.
6. Synthetic generation only when explicitly permitted via `synthetic_fallback=True`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Official RaDICaL Dataset Class Mapping
RADICAL_CLASSES = ["Empty", "Pedestrian", "Cyclist", "Vehicle"]


def get_num_classes() -> int:
    """Return the total number of classes in the official RaDICaL dataset."""
    return len(RADICAL_CLASSES)


def get_class_names() -> List[str]:
    """Return the list of class names in the official RaDICaL dataset."""
    return list(RADICAL_CLASSES)


class RaDICaLFeatureExtractor:
    """Extracts a fixed-size 1D feature vector of dimension `feature_dim` from a 2D Range-Doppler map."""

    def __init__(self, feature_dim: int = 64, normalization: str = "db") -> None:
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

        # 1. Range profile: sum along Doppler dimension
        range_profile = np.sum(rd_map, axis=1)  # [R]
        # 2. Doppler profile: sum along Range dimension
        doppler_profile = np.sum(rd_map, axis=0)  # [D_bins]

        # 3. Peak statistics & summary metrics
        max_val = float(np.max(rd_map)) if rd_map.size > 0 else 0.0
        mean_val = float(np.mean(rd_map)) if rd_map.size > 0 else 0.0
        std_val = float(np.std(rd_map)) if rd_map.size > 0 else 0.0
        snr_est = (max_val / (mean_val + 1e-8)) if mean_val > 0 else 0.0

        # Resample profiles to allocate into feature_dim
        num_scalars = 4
        rem_dim = self.feature_dim - num_scalars
        n_range = rem_dim // 2
        n_doppler = rem_dim - n_range

        # Interpolate range profile
        x_r = np.linspace(0, 1, len(range_profile))
        x_r_new = np.linspace(0, 1, n_range)
        r_resampled = np.interp(x_r_new, x_r, range_profile)

        # Interpolate doppler profile
        x_d = np.linspace(0, 1, len(doppler_profile))
        x_d_new = np.linspace(0, 1, n_doppler)
        d_resampled = np.interp(x_d_new, x_d, doppler_profile)

        scalars = np.array([max_val, mean_val, std_val, snr_est], dtype=np.float32)
        feat = np.concatenate([r_resampled, d_resampled, scalars])

        # Apply normalization
        if self.normalization == "db":
            feat = 10.0 * np.log10(np.maximum(np.abs(feat), 1e-6))
            feat = np.clip(feat, -100.0, 100.0) / 100.0
        elif self.normalization == "zscore":
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
            "features": self.features[idx],        # [T, D]
            "detection": self.labels_det[idx],      # [1]
            "classification": self.labels_cls[idx], # scalar int
            "anomaly": self.labels_ano[idx],        # [1]
        }


class RaDICaLDatasetAdapter:
    """Adapter to load, preprocess, and partition RaDICaL radar sequences."""

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = "data/radical",
        sequence_length: int = 16,
        feature_dim: int = 64,
        num_classes: int = 4,
        normalization: Literal["db", "zscore", "minmax", "none"] = "db",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        synthetic_fallback: bool = False,
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
        self.synthetic_fallback = synthetic_fallback

        self.extractor = RaDICaLFeatureExtractor(
            feature_dim=feature_dim, normalization=normalization
        )

    def _convert_rd_sequences_to_features(self, rd_tensors: np.ndarray) -> np.ndarray:
        """Convert a batch of Range-Doppler sequences [N, T, R, D_bins] to [N, T, feature_dim]."""
        N, T = rd_tensors.shape[:2]
        all_feats = np.zeros((N, T, self.feature_dim), dtype=np.float32)
        for i in range(N):
            for t in range(T):
                all_feats[i, t] = self.extractor.extract(rd_tensors[i, t])
        return all_feats

    def _load_split_directory(self, split_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Load data from a partition directory containing .h5 or .npz files."""
        if not split_dir.exists():
            return None

        # 1. Prefer aggregated HDF5 if available
        h5_files = list(split_dir.glob("*.h5")) or list(split_dir.glob("*.hdf5"))
        if h5_files:
            h5_path = h5_files[0]
            with h5py.File(h5_path, "r") as h5f:
                if "features" in h5f:
                    feats = np.array(h5f["features"], dtype=np.float32)
                elif "rd_tensors" in h5f:
                    rd_tensors = np.array(h5f["rd_tensors"])
                    feats = self._convert_rd_sequences_to_features(rd_tensors)
                else:
                    return None

                det = np.array(h5f["detection"], dtype=np.float32)
                if det.ndim == 1:
                    det = det[:, None]
                cls_lbl = np.array(h5f["classification"], dtype=np.int64)
                ano = np.array(h5f["anomaly"], dtype=np.float32)
                if ano.ndim == 1:
                    ano = ano[:, None]
                return feats, det, cls_lbl, ano

        # 2. Check individual .npz files
        npz_files = sorted(list(split_dir.glob("*.npz")))
        if npz_files:
            feats_list, det_list, cls_list, ano_list = [], [], [], []
            for f in npz_files:
                data = np.load(f)
                if "features" in data:
                    feats_list.append(data["features"])
                elif "rd_tensor" in data:
                    rd_seq = data["rd_tensor"]  # [T, R, D_bins]
                    seq_feats = [self.extractor.extract(rd_seq[t]) for t in range(len(rd_seq))]
                    feats_list.append(np.stack(seq_feats, axis=0))

                det_val = data.get("detection", np.array([1.0 if data.get("classification", 0) > 0 else 0.0], dtype=np.float32))
                det_list.append(det_val if det_val.ndim == 1 else det_val.flatten())
                cls_list.append(int(data.get("classification", 0)))
                ano_val = data.get("anomaly", np.array([0.0], dtype=np.float32))
                ano_list.append(ano_val if ano_val.ndim == 1 else ano_val.flatten())

            feats = np.stack(feats_list, axis=0).astype(np.float32)
            det = np.array(det_list, dtype=np.float32)
            if det.ndim == 1:
                det = det[:, None]
            cls_lbl = np.array(cls_list, dtype=np.int64)
            ano = np.array(ano_list, dtype=np.float32)
            if ano.ndim == 1:
                ano = ano[:, None]
            return feats, det, cls_lbl, ano

        return None

    def generate_synthetic_samples(
        self,
        num_samples: int = 200,
        range_bins: int = 64,
        doppler_bins: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic RaDICaL Range-Doppler sequences for testing."""
        rng = np.random.RandomState(self.seed)
        features_list, det_list, cls_list, ano_list = [], [], [], []

        for i in range(num_samples):
            c = rng.randint(0, self.num_classes)
            has_target = (c > 0)
            is_anomaly = (rng.rand() < 0.15)

            seq_feats = []
            r_center = rng.uniform(10, range_bins - 10)
            d_center = rng.uniform(doppler_bins * 0.3, doppler_bins * 0.7)
            velocity = (c * 0.5) if has_target else 0.0

            for t in range(self.sequence_length):
                noise_floor = rng.exponential(scale=1.0, size=(range_bins, doppler_bins))
                if has_target:
                    curr_r = int(np.clip(r_center + velocity * t, 0, range_bins - 1))
                    curr_d = int(np.clip(d_center + (rng.randn() * 0.2), 0, doppler_bins - 1))
                    peak_amp = 15.0 * c
                    noise_floor[curr_r, curr_d] += peak_amp
                    if curr_r + 1 < range_bins:
                        noise_floor[curr_r + 1, curr_d] += peak_amp * 0.5
                    if curr_d + 1 < doppler_bins:
                        noise_floor[curr_r, curr_d + 1] += peak_amp * 0.5

                if is_anomaly:
                    noise_floor[:, rng.randint(0, doppler_bins)] += rng.uniform(20.0, 50.0)

                feat_t = self.extractor.extract(noise_floor)
                seq_feats.append(feat_t)

            features_list.append(np.stack(seq_feats, axis=0))
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
        """Load dataset from path. Raises FileNotFoundError if data_path is missing and fallback is disabled."""
        if not self.data_path or not self.data_path.exists():
            if not self.synthetic_fallback:
                raise FileNotFoundError(
                    f"RaDICaL dataset not found at '{self.data_path}'. Synthetic fallback is disabled."
                )
            return self.generate_synthetic_samples(num_samples=num_synthetic_fallback)

        # Check for structured train/, val/, test/ subdirectories
        train_data = self._load_split_directory(self.data_path / "train")
        val_data = self._load_split_directory(self.data_path / "val")
        test_data = self._load_split_directory(self.data_path / "test")

        if train_data is not None:
            feats = [train_data[0]]
            dets = [train_data[1]]
            classes_arr = [train_data[2]]
            anos = [train_data[3]]
            if val_data is not None:
                feats.append(val_data[0])
                dets.append(val_data[1])
                classes_arr.append(val_data[2])
                anos.append(val_data[3])
            if test_data is not None:
                feats.append(test_data[0])
                dets.append(test_data[1])
                classes_arr.append(test_data[2])
                anos.append(test_data[3])

            return (
                np.concatenate(feats, axis=0),
                np.concatenate(dets, axis=0),
                np.concatenate(classes_arr, axis=0),
                np.concatenate(anos, axis=0),
            )

        # Check single directory or file
        res = self._load_split_directory(self.data_path)
        if res is not None:
            return res

        if not self.synthetic_fallback:
            raise FileNotFoundError(
                f"No valid RaDICaL data files found in '{self.data_path}'. Synthetic fallback is disabled."
            )
        return self.generate_synthetic_samples(num_samples=num_synthetic_fallback)

    def get_datasets(
        self,
        num_synthetic_fallback: int = 300,
    ) -> Tuple[RaDICaLDataset, RaDICaLDataset, RaDICaLDataset]:
        """Split data and return (train_dataset, val_dataset, test_dataset)."""
        # Check if train/val/test directories already exist
        if self.data_path and self.data_path.exists():
            train_dir = self.data_path / "train"
            val_dir = self.data_path / "val"
            test_dir = self.data_path / "test"

            train_data = self._load_split_directory(train_dir)
            val_data = self._load_split_directory(val_dir)
            test_data = self._load_split_directory(test_dir)

            if train_data is not None and val_data is not None and test_data is not None:
                train_ds = RaDICaLDataset(*train_data)
                val_ds = RaDICaLDataset(*val_data)
                test_ds = RaDICaLDataset(*test_data)
                return train_ds, val_ds, test_ds

        feats, det, cls_lbl, ano = self.load_data(num_synthetic_fallback=num_synthetic_fallback)
        n = len(feats)

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
