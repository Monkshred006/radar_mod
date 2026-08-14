"""Inspect and Validate RaDICaL Dataset Labels and Taxonomy.

Enumerates labels across train/val/test splits, checks class balance,
verifies taxonomy alignment against {0: 'Empty', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Vehicle'},
and saves a comprehensive report to `results/photon_v0/radical_label_report.json`.
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
import yaml

from module_03_sensor_fusion.radical_adapter import (
    RaDICaLDatasetAdapter,
    get_num_classes,
    get_class_names,
    RADICAL_CLASSES,
)

EXPECTED_TAXONOMY = {
    0: "Empty",
    1: "Pedestrian",
    2: "Cyclist",
    3: "Vehicle",
}


def inspect_labels(
    dataset_path: Union[str, Path] = "data/radical",
    config_path: Union[str, Path] = "configs/photon_v0.yaml",
    output_report_path: Union[str, Path] = "results/photon_v0/radical_label_report.json",
) -> Dict[str, Any]:
    """Inspect dataset labels and produce verification report."""
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"RaDICaL dataset directory '{dataset_dir}' does not exist.")

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    adapter = RaDICaLDatasetAdapter(
        data_path=dataset_dir,
        synthetic_fallback=False,
    )
    train_ds, val_ds, test_ds = adapter.get_datasets()

    splits_data = {
        "train": train_ds.labels_cls.numpy(),
        "val": val_ds.labels_cls.numpy(),
        "test": test_ds.labels_cls.numpy(),
    }

    all_labels = np.concatenate(list(splits_data.values()))
    unique_labels = sorted(list(np.unique(all_labels)))
    total_samples = len(all_labels)

    # Class frequency counts
    overall_counts = {}
    split_counts = {"train": {}, "val": {}, "test": {}}

    for c in unique_labels:
        c_int = int(c)
        c_name = RADICAL_CLASSES[c_int] if c_int < len(RADICAL_CLASSES) else f"Class_{c_int}"
        cnt = int(np.sum(all_labels == c_int))
        pct = (cnt / total_samples) * 100.0 if total_samples > 0 else 0.0
        overall_counts[c_int] = {
            "class_name": c_name,
            "count": cnt,
            "percentage": round(pct, 2),
        }

        for split_name, split_arr in splits_data.items():
            s_cnt = int(np.sum(split_arr == c_int))
            split_counts[split_name][c_int] = s_cnt

    # Verify taxonomy match
    taxonomy_matches = (
        len(unique_labels) == len(EXPECTED_TAXONOMY)
        and all(c in EXPECTED_TAXONOMY for c in unique_labels)
    )

    # If mismatch, update configs/photon_v0.yaml automatically
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if cfg.get("dataset", {}).get("num_classes") != len(unique_labels):
            print(f"[RaDICaL Label Inspector] Updating config num_classes to {len(unique_labels)}")
            cfg["dataset"]["num_classes"] = len(unique_labels)
            cfg["model"]["num_classes"] = len(unique_labels)
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False)

    report = {
        "dataset_path": str(dataset_dir.absolute()),
        "total_samples": total_samples,
        "unique_labels": [int(x) for x in unique_labels],
        "num_classes": len(unique_labels),
        "taxonomy_verified": taxonomy_matches,
        "expected_taxonomy": {str(k): v for k, v in EXPECTED_TAXONOMY.items()},
        "observed_class_distribution": {str(k): v for k, v in overall_counts.items()},
        "split_distribution": {
            s: {str(k): v for k, v in sc.items()} for s, sc in split_counts.items()
        },
        "sample_split_totals": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Formatted Frequency Table
    print("================================================================")
    print(" RaDICaL Dataset Label & Taxonomy Verification Report")
    print("================================================================")
    print(f"Dataset Path:  {dataset_dir.absolute()}")
    print(f"Total Samples: {total_samples} (Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)})")
    print(f"Taxonomy Match: {'VERIFIED (Matches Standard)' if taxonomy_matches else 'MISMATCH'}")
    print("----------------------------------------------------------------")
    print(f"{'Class ID':<10} {'Class Name':<15} {'Total Count':<12} {'Percentage':<12} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("----------------------------------------------------------------")
    for c_int in unique_labels:
        stats = overall_counts[c_int]
        t_cnt = split_counts["train"].get(c_int, 0)
        v_cnt = split_counts["val"].get(c_int, 0)
        te_cnt = split_counts["test"].get(c_int, 0)
        print(f"{c_int:<10} {stats['class_name']:<15} {stats['count']:<12} {stats['percentage']:>5.1f}%       {t_cnt:<8} {v_cnt:<8} {te_cnt:<8}")
    print("================================================================")
    print(f"Report saved to: {out_file.absolute()}\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RaDICaL dataset labels.")
    parser.add_argument("--data-path", type=str, default="data/radical", help="Path to RaDICaL dataset")
    parser.add_argument("--config", type=str, default="configs/photon_v0.yaml", help="Path to YAML config")
    parser.add_argument("--output", type=str, default="results/photon_v0/radical_label_report.json", help="Report path")
    args = parser.parse_args()

    inspect_labels(dataset_path=args.data_path, config_path=args.config, output_report_path=args.output)


if __name__ == "__main__":
    main()
