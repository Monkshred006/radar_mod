# Module 2 — Sensor Signal Preprocessing / DSP

## 1. Purpose

Module 2 performs all **signal preprocessing** between raw sensor measurements (Module 1) and the sensor fusion / feature extraction stage (Module 3). It does **not** implement any AI, classification, or decision logic.

---

## 2. Relationship with Module 1

Module 1 exposes a `RadarDataset` whose `__getitem__` returns:
```python
{
    "radar":     torch.Tensor,   # shape [T, ...], original dtype
    "timestamp": torch.Tensor,   # shape [T], float64 seconds
    "metadata":  dict
}
```
Module 2 consumes this via `SensorDSPPipeline.from_module1_sample(sample, channel_names=[...])` which bridges Module 1's generic tensor schema to Module 2's named-channel dict.

**Module 1 is not modified.**

---

## 3. Supported PhotonShield Sensors

| Channel | Sensor | Notes |
|---|---|---|
| `photodiode_1`, `photodiode_2` | BPW34 | Primary optical sensing |
| `temperature`, `humidity`, `pressure` | BME280 | Environmental |
| `accel_x/y/z`, `gyro_x/y/z` | MPU6050 | IMU |
| `distance` | VL53L0X | Time-of-flight |

Missing sensors are **never silently fabricated**. Absent channels are tracked as missing in validity metadata.

---

## 4. Filtering Options

| Filter | Key | Notes |
|---|---|---|
| Moving average | `"moving_average"` | Causal in streaming, centred offline |
| Exponential MA | `"ema"` | Always causal |
| Median | `"median"` | Causal in streaming |
| Butterworth LP | `"lowpass"` | `filtfilt` offline (zero-phase), `lfilter` streaming |
| None | `"none"` | Pass-through |

---

## 5. Synchronization

Multi-rate streams are aligned to a **uniform target grid** at `sync.target_rate_hz`.

| Mode | Allowed methods |
|---|---|
| Offline | `"nearest"`, `"ffill"`, `"linear"` |
| Streaming | `"nearest"`, `"ffill"` (**causal only**) |

Gaps > `sync.max_gap_s` are left as `NaN` and flagged as missing.

---

## 6. Missing Data Handling

Strategies (configurable per channel):
- `"leave"` — keep NaN as-is
- `"ffill"` — forward-fill / zero-order hold (causal, streaming-safe)
- `"interpolate"` — linear interpolation (offline only; falls back to ffill in streaming)
- `"constant"` — fill with `fill_value`

Missing samples are **always tracked** in validity masks.

---

## 7. Outlier Handling

Outliers are **flagged, not removed**. Methods: IQR, Z-score, MAD, physical range bounds.

---

## 8. Normalization

`SensorScaler` supports `"minmax"`, `"standard"` (z-score), `"robust"` (median/IQR).

**Anti-leakage rule**: `fit()` on training data → `transform()` on val/test. Statistics serializable to JSON.

---

## 9. Signal Quality Metrics (per channel)

`missing_pct`, `interpolated_pct`, `outlier_pct`, `nan_pct`, `mean`, `std`, `variance`, `signal_range`, `saturation_pct`, `timestamp_jitter_mean_s`, `timestamp_jitter_max_s`.

---

## 10. Offline Processing

```python
from module_02_sensor_dsp import SensorDSPPipeline, SensorDSPConfig

config = SensorDSPConfig()
pipe = SensorDSPPipeline(config)

raw_data = {
    "timestamps": {"photodiode_1": ts_array, "temperature": ts_env},
    "values":     {"photodiode_1": pd_array, "temperature": temp_array},
}

pipe.fit_scalers(raw_data)       # fit on training data
result = pipe.process_offline(raw_data)

signals = result["signals"]      # dict channel -> float64 array
quality = result["quality"]      # dict channel -> metrics dict
```

---

## 11. Streaming Processing

```python
state = pipe.make_stream_state()
for t, sample in sensor_stream:
    output, state = pipe.process_stream(sample, state, tgt_time=t)
    signal_vals = output["signals"]  # dict channel -> float
```

**Streaming enforces causal processing** — no future samples are accessed.

---

## 12. Module 1 → Module 2 Bridge

```python
from module_01_radar_input import RadarDataset, RadarDatasetConfig
from module_02_sensor_dsp import SensorDSPPipeline

dataset = RadarDataset(RadarDatasetConfig(dataset_path="..."))
sample = dataset[0]

raw_data = SensorDSPPipeline.from_module1_sample(
    sample, channel_names=["photodiode_1", "photodiode_2"]
)
result = pipe.process_offline(raw_data)
```

---

## 13. Output Interface for Module 3

```python
{
    "signals": {channel: np.ndarray},        # float64, synchronized, processed
    "timestamps": np.ndarray,                # float64 unified grid
    "validity": {
        "outlier_masks":      {channel: bool array},
        "missing_masks":      {channel: bool array},
        "interpolated_masks": {channel: bool array},
    },
    "quality": {channel: {metrics dict}},
    "preprocessing_metadata": {
        "channels_processed": [...],
        "sync_rate_hz": float,
        "mode": "offline" | "streaming",
        "normalization_states": {channel: {state dict}},
    }
}
```

---

## 14. Pipeline Diagram

```
Raw sensor data
      ↓
Validation (NaN/Inf/type checks)
      ↓
Timestamp Synchronization (multi-rate → uniform grid)
      ↓
Missing Data Handling (ffill / interpolate / leave)
      ↓
Baseline Correction (photodiode-specific, optional per channel)
      ↓
Filtering / Denoising (MA / EMA / Median / Lowpass)
      ↓
Outlier Detection (flag only — IQR / z-score / MAD / range)
      ↓
Normalization (minmax / standard / robust — training stats only)
      ↓
Signal Quality Computation
      ↓
Processed synchronized sensor streams
      ↓
Module 3 (Sensor Fusion + Feature Extraction)
```
