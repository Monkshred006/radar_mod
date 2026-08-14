# Module 3 — Sensor Fusion + Feature Extraction

## 1. Purpose

Module 3 converts synchronized, cleaned sensor streams (Module 2) into structured temporal feature matrices and **sensor-aware 4D tokens** for the downstream Mamba-Hybrid Engine (Module 4).

**Strict Boundary**: Module 3 is completely deterministic and contains **no neural layers, trainable weights, attention, or BitNet quantization**.

---

## 2. Relationships with Modules 1 & 2

```
Module 1 (Sensor Data Ingestion)
      ↓
Module 2 (Sensor DSP / Preprocessing)
      ↓
Module 3 (Sensor Fusion & Feature Extraction)
      ↓
Module 4 (Mamba-Hybrid Engine)
```

Module 3 consumes Module 2's `ProcessedOutput` dictionary directly:
```python
{
    "signals":    {"photodiode_1": np.ndarray [T], ...},
    "timestamps": np.ndarray [T],
    "validity":   {"outlier_masks": {...}, "missing_masks": {...}, ...},
    "quality":    {channel_name: metrics_dict},
}
```

---

## 3. Sensor Groups

| Group Name | Sensors / Channels | Features Extracted |
|---|---|---|
| `"optical"` | `photodiode_1`, `photodiode_2` (BPW34) | Amplitude, $\Delta x$, $dx/dt$, rolling mean, rolling std, energy |
| `"environment"` | `temperature`, `humidity`, `pressure` (BME280) | Current value, $\Delta x$, rolling mean |
| `"motion"` | `accel_x/y/z`, `gyro_x/y/z` (MPU6050) | Raw X/Y/Z, $\Delta x$, rolling mean, rolling std, `accel_magnitude`, `gyro_magnitude` |
| `"distance"` | `distance` (VL53L0X) | Current value, $\Delta x$, $dx/dt$, rolling mean |
| `"quality"` | All channels | `is_valid`, `is_outlier`, `is_missing`, `is_interpolated` flags |

---

## 4. Sensor-Aware Token Representation

Module 3 outputs tokens shaped:
$$\mathbf{Tokens} \in \mathbb{R}^{B \times T \times S \times D_{\text{features}}}$$

where:
- **`B`**: Batch size
- **`T`**: Sequence length (temporal dimension preserved)
- **`S`**: Number of sensor groups (e.g. 5: optical, env, motion, distance, quality)
- **`D_features`**: Feature vector dimension per sensor group (padded to max $D$)

Module 3 also provides a boolean mask $\mathbf{Mask} \in \{0, 1\}^{B \times T \times S \times D_{\text{features}}}$ indicating valid vs padded feature positions.

**Decoupled Model Dimension**: $D_{\text{features}}$ is explicitly decoupled from Mamba's internal model dimension $D_{\text{model}}$ (Module 4 performs the learnable projection).

---

## 5. Temporal Preservation & Causality

- **No sequence collapsing**: $T$ is strictly preserved (`[B, T, F_fused]` and `[B, T, S, D_features]`).
- **Streaming Causality**: All rolling features ($\Delta x_t = x_t - x_{t-1}$, rolling mean/std/energy) use causal history only (no future lookahead).

---

## 6. Code Example

```python
from module_02_sensor_dsp import SensorDSPPipeline, SensorDSPConfig
from module_03_sensor_fusion import SensorFusionPipeline, Module3Config

# 1. Run Module 2 DSP
dsp_pipe = SensorDSPPipeline()
dsp_out = dsp_pipe.process_offline(raw_sensor_data)

# 2. Run Module 3 Fusion & Tokenization
fusion_pipe = SensorFusionPipeline()
fusion_out = fusion_pipe.process_offline(dsp_out)

tokens = fusion_out["tokens"]          # torch.Tensor [T, S, D_features]
token_mask = fusion_out["token_mask"]  # torch.Tensor [T, S, D_features]
fused_feat = fusion_out["features"]    # torch.Tensor [T, F_fused]
```

---

## 7. Architecture Diagram

```
Module 1
Raw Sensor Data
      ↓
Module 2
Clean Synchronized Signals + Validity Masks
      ↓
Module 3 — Sensor Fusion + Feature Extraction
┌──────────────────────────────────────────────┐
│ Sensor Grouping                              │
│ (Optical, Environment, Motion, Distance)     │
│        ↓                                     │
│ Feature Extraction                           │
│ (Amplitude, Mags, Diffs, Rolling, Quality)   │
│        ↓                                     │
│ Deterministic Feature Fusion                 │
│ [B, T, F_fused]                              │
│        ↓                                     │
│ Sensor-Aware Tokenization                    │
│ [B, T, S, D_features]                        │
└──────────────────────┬───────────────────────┘
                       ↓
         Module 4 — Mamba-Hybrid Engine
```
