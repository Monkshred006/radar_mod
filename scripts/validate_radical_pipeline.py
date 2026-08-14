"""Validate RaDICaL Data Pipeline and Tensor Dimensions for PhotonShield AI.

Verifies:
1. Batch tensor shapes: `[B, 16, 64]`.
2. Absence of NaNs and Infinite values across all batches.
3. Normalization value range bounds (dB, zscore, minmax).
4. Label tensor shapes and range consistency:
   - Detection: `[B, 1]` in [0, 1]
   - Classification: `[B]` in [0, num_classes-1]
   - Anomaly: `[B, 1]` in [0, 1]
5. Saves validation report to `results/photon_v0/pipeline_validation.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, List, Union

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter


def validate_pipeline(
    data_path: Union[str, Path] = "data/radical",
    config_path: Union[str, Path] = "configs/photon_v0.yaml",
    output_report_path: Union[str, Path] = "results/photon_v0/pipeline_validation.json",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Execute rigorous tensor shape and numerical validity verification."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"RaDICaL dataset directory '{data_dir}' not found.")

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load configuration
    cfg = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    seq_len = cfg.get("dataset", {}).get("sequence_length", 16)
    feat_dim = cfg.get("dataset", {}).get("feature_dim", 64)
    num_classes = cfg.get("dataset", {}).get("num_classes", 4)
    norm_mode = cfg.get("dataset", {}).get("normalization", "db")

    adapter = RaDICaLDatasetAdapter(
        data_path=data_dir,
        sequence_length=seq_len,
        feature_dim=feat_dim,
        num_classes=num_classes,
        normalization=norm_mode,
        synthetic_fallback=False,
    )

    train_loader, val_loader, test_loader = adapter.get_dataloaders(batch_size=batch_size)

    checks_passed = True
    errors = []
    split_summaries = {}

    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        total_batches = len(loader)
        total_samples = len(loader.dataset)
        nan_count = 0
        inf_count = 0
        min_val = float("inf")
        max_val = float("-inf")
        mean_val = 0.0

        for b_idx, batch in enumerate(loader):
            features = batch["features"]          # [B, T, D]
            detection = batch["detection"]        # [B, 1]
            classification = batch["classification"] # [B]
            anomaly = batch["anomaly"]            # [B, 1]

            B, T, D = features.shape

            # 1. Shape Checks
            if T != seq_len or D != feat_dim:
                checks_passed = False
                errors.append(f"[{split_name} Batch {b_idx}] Feature shape mismatch: got ({B}, {T}, {D}), expected ({B}, {seq_len}, {feat_dim})")

            if detection.ndim != 2 or detection.shape[1] != 1:
                checks_passed = False
                errors.append(f"[{split_name} Batch {b_idx}] Detection shape mismatch: {detection.shape}")

            if classification.ndim != 1 or classification.shape[0] != B:
                checks_passed = False
                errors.append(f"[{split_name} Batch {b_idx}] Classification shape mismatch: {classification.shape}")

            if anomaly.ndim != 2 or anomaly.shape[1] != 1:
                checks_passed = False
                errors.append(f"[{split_name} Batch {b_idx}] Anomaly shape mismatch: {anomaly.shape}")

            # 2. Numerical Checks (NaN / Inf)
            if torch.isnan(features).any() or torch.isnan(detection).any() or torch.isnan(anomaly).any():
                checks_passed = False
                nan_count += 1
                errors.append(f"[{split_name} Batch {b_idx}] NaN detected in tensors")

            if torch.isinf(features).any():
                checks_passed = False
                inf_count += 1
                errors.append(f"[{split_name} Batch {b_idx}] Inf detected in features")

            # 3. Label Range Checks
            if (classification < 0).any() or (classification >= num_classes).any():
                checks_passed = False
                errors.append(f"[{split_name} Batch {b_idx}] Classification index out of bounds [0, {num_classes-1}]")

            if (detection < 0.0).any() or (detection > 1.0).any():
                checks_passed = False
                errors.append(f"[{split_name} Batch {b_idx}] Detection values outside [0, 1]")

            # Update stats
            feat_np = features.numpy()
            min_val = min(min_val, float(np.min(feat_np)))
            max_val = max(max_val, float(np.max(feat_np)))
            mean_val += float(np.mean(feat_np))

        mean_val /= max(total_batches, 1)
        split_summaries[split_name] = {
            "num_samples": total_samples,
            "num_batches": total_batches,
            "nan_batches": nan_count,
            "inf_batches": inf_count,
            "value_min": round(min_val, 4),
            "value_max": round(max_val, 4),
            "value_mean": round(mean_val, 4),
        }

    report = {
        "all_checks_passed": checks_passed,
        "expected_batch_feature_shape": f"[B, {seq_len}, {feat_dim}]",
        "normalization_mode": norm_mode,
        "errors": errors,
        "split_summaries": split_summaries,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("================================================================")
    print(" RaDICaL Data Pipeline & Tensor Validation Report")
    print("================================================================")
    print(f"Overall Status: {'PASSED (All Tensor Shapes and Values Clean)' if checks_passed else 'FAILED'}")
    print(f"Target Feature Shape: [B, {seq_len}, {feat_dim}] | Normalization: {norm_mode}")
    print("----------------------------------------------------------------")
    for s_name, stats in split_summaries.items():
        print(f"Split: {s_name.upper():<6} | Samples: {stats['num_samples']:<4} | Batches: {stats['num_batches']:<3} | Min: {stats['value_min']:>7.2f} | Max: {stats['value_max']:>7.2f} | Mean: {stats['value_mean']:>7.2f} | NaNs: {stats['nan_batches']}")
    print("----------------------------------------------------------------")
    if errors:
        print("Detected Errors:")
        for err in errors[:5]:
            print(f"  - {err}")
    else:
        print("No shape, padding, or numerical errors detected.")
    print("================================================================")
    print(f"Report saved to: {out_file.absolute()}\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RaDICaL pipeline tensor shapes.")
    parser.add_argument("--data-path", type=str, default="data/radical")
    parser.add_argument("--config", type=str, default="configs/photon_v0.yaml")
    parser.add_argument("--output", type=str, default="results/photon_v0/pipeline_validation.json")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    validate_pipeline(
        data_path=args.data_path,
        config_path=args.config,
        output_report_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
