# Module 1: Radar Data Input / Data Ingestion

## 1. Purpose
Module 1 provides a clean, modular radar data ingestion pipeline. It is responsible solely for loading, validating, organizing, sequence construction, scene-level splitting, and PyTorch `DataLoader` batching of raw radar datasets.

**Important**: This module performs **NO** radar DSP (FFT, Range-Doppler, Angle estimation, CFAR) or neural network normalization/FP16 conversion.

---

## 2. Expected Dataset Structure
The default loader (`DefaultDirectoryAdapter`) supports directory hierarchies structured by scene:

```
dataset_root/
├── scene_01/
│   ├── frame_000.npy
│   ├── frame_001.npy
│   └── ...
├── scene_02/
│   ├── frame_000.npz
│   └── ...
```
Or flat directories containing files with arbitrary extension supported by `RadarLoader`.

---

## 3. Supported File Formats
- NumPy arrays (`.npy`)
- Compressed NumPy archives (`.npz`)
- CSV tables (`.csv`)
- JSON data arrays (`.json`)

All formats preserve the original numerical representation (e.g. `complex64`, `float32`, `int16`).

---

## 4. Configuration
Configured via `RadarDatasetConfig` (Python dataclass or YAML):
- `dataset_path`: Directory path containing radar frame files.
- `sequence_length`: Number of frames per temporal sequence ($T$).
- `frame_stride`: Step size between consecutive frames *within* a sequence.
- `sequence_stride`: Step size between start frames of consecutive sequences.
- `train_ratio`, `val_ratio`, `test_ratio`: Ratios for scene-level splitting.
- `batch_size`, `num_workers`, `shuffle`, `random_seed`.

---

## 5. Temporal Sequence Construction
Given sequence length $T=16$, frame stride $FS=1$, sequence stride $SS=1$:
- Sequence 0: `[Frame 0, Frame 1, ..., Frame 15]`
- Sequence 1: `[Frame 1, Frame 2, ..., Frame 16]`

If $FS=2$:
- Sequence 0: `[Frame 0, Frame 2, Frame 4, ..., Frame 30]`

---

## 6. Output Format Interface
Output from `RadarDataset[i]` returns a structured Python dictionary:
```python
{
    "radar": torch.Tensor,       # Shape: [T, ...], preserving original dtype (e.g. float32, complex64)
    "timestamp": torch.Tensor,   # Shape: [T]
    "metadata": {
        "scene_id": str,
        "sequence_id": str,
        "frame_metadata": [...]
    }
}
```

---

## 7. Scene-Level Splitting & Anti-Leakage
Dataset splitting is performed at the **SCENE level** (`scene_id`). All sequences generated from the same scene remain strictly in the same split (Train / Val / Test) to eliminate temporal/spatial leakage across training splits. Split assignments and random seed are stored reproducible in `split_info.json`.

---

## 8. How Module 2 Consumes Output
Module 2 (Radar DSP) will instantiate `RadarDataset` and receive `[B, T, ...]` batched tensors directly from `DataLoader`:

```python
from module_01_radar_input import RadarDatasetConfig, RadarDataset, create_dataloaders, split_dataset

config = RadarDatasetConfig(dataset_path="path/to/dataset", sequence_length=16, batch_size=8)
dataset = RadarDataset(config)
train_ds, val_ds, test_ds = split_dataset(dataset, output_dir="runs/manifest")
train_loader, _, _ = create_dataloaders(train_ds, val_ds, test_ds)

for batch in train_loader:
    raw_radar = batch["radar"] # Shape: [B, T, ...]
    # Pass raw_radar to Module 2 (DSP processing)...
```

---

## 9. Dataset Inspection CLI
Inspect dataset metadata, sequence count, and split breakdown without training:
```bash
python -m module_01_radar_input.inspect --dataset-path path/to/dataset
```
