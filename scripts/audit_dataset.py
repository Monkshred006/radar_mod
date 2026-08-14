"""Deep audit of the RaDICaL dataset directory and sample limits."""

from __future__ import annotations

import json
from pathlib import Path
import h5py
import numpy as np

DATA_DIR = Path("C:/Users/worka/research/photonpinn/data/radical")

def audit_dataset():
    print("================================================================")
    print(" RaDICaL Dataset Deep Audit Report")
    print("================================================================")
    print(f"Dataset root: {DATA_DIR.absolute()}")
    print(f"Exists: {DATA_DIR.exists()}")

    all_files = [f for f in DATA_DIR.rglob("*") if f.is_file()]
    print(f"Total files: {len(all_files)}")

    file_types = {}
    for f in all_files:
        ext = f.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    print(f"File types: {file_types}")

    total_frames = 0
    total_sequences = 0
    split_info = {}

    for split in ["train", "val", "test"]:
        split_p = DATA_DIR / split
        npz_files = list(split_p.glob("*.npz"))
        h5_files = list(split_p.glob("*.h5"))
        
        split_seqs = 0
        split_cls_dist = {}
        
        # Check H5 master archive
        h5_file = split_p / f"{split}_radical.h5"
        if h5_file.exists():
            with h5py.File(h5_file, "r") as h5f:
                if "rd_tensors" in h5f:
                    s_shape = h5f["rd_tensors"].shape  # [N, T, R, D]
                    split_seqs = s_shape[0]
                    total_sequences += s_shape[0]
                    total_frames += s_shape[0] * s_shape[1]
                if "classification" in h5f:
                    cls_arr = np.array(h5f["classification"])
                    unq, cnts = np.unique(cls_arr, return_counts=True)
                    split_cls_dist = {int(k): int(v) for k, v in zip(unq, cnts)}
        else:
            split_seqs = len(npz_files)
            total_sequences += split_seqs
            total_frames += split_seqs * 16

        split_info[split] = {
            "npz_count": len(npz_files),
            "h5_count": len(h5_files),
            "total_sequences": split_seqs,
            "class_distribution": split_cls_dist,
        }
        print(f"Split [{split.upper()}]: {split_seqs} sequences ({len(npz_files)} .npz, {len(h5_files)} .h5), distribution: {split_cls_dist}")

    print(f"Total Dataset Sequences: {total_sequences}")
    print(f"Total Radar Frames:      {total_frames}")

    # Inspect labels
    class_map_file = DATA_DIR / "labels" / "class_mapping.json"
    if class_map_file.exists():
        with open(class_map_file, "r") as f:
            class_map = json.load(f)
        print(f"Class Mapping: {class_map}")

    manifest_file = DATA_DIR / "labels" / "label_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
        print(f"Manifest total registered samples: {len(manifest)}")

    print("================================================================")
    return {
        "total_files": len(all_files),
        "file_types": file_types,
        "total_sequences": total_sequences,
        "total_frames": total_frames,
        "split_info": split_info,
    }

if __name__ == "__main__":
    audit_dataset()
