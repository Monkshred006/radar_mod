# Module 5 — FP32 Training + Evaluation Pipeline

## Purpose

Module 5 is the first complete training system for the PhotonShield AI Mamba-Hybrid model. It provides:

- **FP32 baseline training** (the reference before any quantization)
- **Target-agnostic training** (configurable for regression, classification, or multi-task)
- **Causally-correct temporal preprocessing** (scene-level Module 2/3 processing)
- **Evaluation pipeline** for held-out test sets
- **Checkpointing** for training resume and FP32 reference preservation
- **Experiment logging** (JSON + CSV, no external services)

## Architecture Position

```
Module 1 — Raw sensor ingestion
    ↓
Module 2 — DSP + synchronization + quality
    ↓
Module 3 — Sensor fusion + feature extraction
    ↓
Module 4 — Mamba-Hybrid Engine
    ↓
Prediction
    ↓
Target Adapter
    ↓
Loss
    ↓
Backpropagation
    ↓
Optimizer
    ↓
Updated Mamba-Hybrid (FP32)
```

## Critical Temporal Causality Constraint

Module 2 and Module 3 have rolling/causal statistics (EMA, rolling windows, baseline correction).

**NEVER apply Module 2/3 per training window in isolation.**

The correct strategy:
1. For each scene, load **all frames in temporal order**
2. Apply `Module2.process_offline(complete_scene_timeline)`
3. Apply `Module3.process_offline(complete_scene_timeline)`
4. **Index windows** from the resulting scene-level feature arrays

This is implemented in `SceneFeatureCache`. Window samples see the correct causal history — no cold-start artifacts.

## FP32 Baseline

```python
mixed_precision = False  # DEFAULT — must not be changed for baseline
```

Training uses standard FP32 PyTorch autograd. No quantization. No AMP. This is the **reference baseline** required before any future BitNet adaptation.

## Data Leakage Prevention

1. **Scene-level split** — `split_dataset()` from Module 1 assigns complete scenes to one split only
2. **No temporal leakage** — windows are indexed from scene timelines; no future frames are used
3. **Normalization leakage prevention** — `SensorDSPPipeline.fit_scalers()` is called ONLY on training scenes; the fitted scaler is applied read-only to val/test

## File Reference

| File | Purpose |
|---|---|
| `config.py` | `TrainingConfig` dataclass |
| `reproducibility.py` | Seed setting + RNG state capture |
| `target_adapter.py` | `TargetAdapter`, synthetic adapters |
| `dataset.py` | `SceneFeatureCache`, `PhotonShieldDataset`, `collate_module3` |
| `losses.py` | Loss factory + `TrainingNaNError` + `WeightedMultiTaskLoss` |
| `metrics.py` | `MetricsTracker` (regression + classification + multi-task) |
| `optimizer.py` | `get_optimizer` factory (AdamW default) |
| `scheduler.py` | `get_scheduler` factory (cosine/step/plateau/none) |
| `early_stopping.py` | `EarlyStopping` (min/max mode) |
| `checkpointing.py` | `save_checkpoint` / `load_checkpoint` |
| `logging_utils.py` | `ExperimentLogger` (JSON + CSV) |
| `trainer.py` | `Trainer.fit()`, gradient management, NaN detection |
| `evaluator.py` | `Evaluator.evaluate()` — test-set only |
| `profiling.py` | `profile_model` — params, latency, throughput |
| `experiment.py` | `ExperimentRunner` — high-level orchestration + ablation |
| `train.py` | CLI: `python -m module_05_training.train --config <path>` |
| `evaluate.py` | CLI: `python -m module_05_training.evaluate --checkpoint <path>` |

## CLI Usage

```bash
# Training
python -m module_05_training.train --config configs/train.json --variant full_mamba_hybrid

# Evaluation
python -m module_05_training.evaluate --checkpoint checkpoints/photonshield_full_mamba_hybrid_best.pt
```

## Configuration

```json
{
  "training": {
    "epochs": 50,
    "learning_rate": 1e-4,
    "batch_size": 16,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "mixed_precision": false,
    "target_type": "regression",
    "num_regression_outputs": 1,
    "random_seed": 42
  },
  "model": {
    "d_model": 128,
    "num_layers": 4,
    "use_mamba": true,
    "use_sensor_attention": true
  }
}
```

## Ablation Variants

Module 5 supports training all three Module 4 ablation configurations:

| Variant | Mamba | Sensor Attention |
|---|---|---|
| `baseline_mamba` | ✅ | ❌ |
| `sensor_interaction_only` | ❌ | ✅ |
| `full_mamba_hybrid` | ✅ | ✅ |

## Future BitNet Transition

Module 5 establishes the **FP32 reference baseline**. The same:
- datasets, splits, target adapter, metrics, and evaluation procedure
will be reused for future BitNet / 1.58-bit model comparison.

Workflow:
```
FP32 Mamba-Hybrid (Module 5)
    → baseline metrics
    → BitNet adaptation (Module 6)
    → 1.58-bit model
    → same evaluation suite
    → fair comparison
```

**DO NOT implement BitNet in Module 5.**

## Running Tests

```bash
cd radar_project
$env:PYTHONPATH = "."
pytest tests/module_05 -v
```

Full suite:
```bash
pytest tests/module_01 tests/module_02 tests/module_03 tests/module_04 tests/module_05 -v
```

## Real Dataset Status

> **IMPORTANT**: No real labeled target dataset is currently defined.
> Module 5 is fully implemented and verified using synthetic deterministic targets.
> Real supervised training will begin once a target dataset and label format are confirmed.
> **Do NOT interpret synthetic test results as real model accuracy.**
