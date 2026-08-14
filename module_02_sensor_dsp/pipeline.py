"""Main preprocessing pipeline for Module 2 — Sensor Signal Preprocessing / DSP.

Integrates: synchronization → filtering → baseline correction → outlier detection
            → missing data handling → normalization → signal quality.

Exposes two primary APIs:
1. process_offline(raw_data) -> ProcessedOutput  [batch mode, all methods allowed]
2. process_stream(sample, state) -> (ProcessedOutput, state)  [causal only]

--- Module 1 → Module 2 Interface ---

Module 1 produces samples with this structure:
    {
        "radar": torch.Tensor,        # shape [T, ...], original dtype
        "timestamp": torch.Tensor,    # shape [T], float64 seconds
        "metadata": {
            "scene_id": str,
            "sequence_id": str,
            "frame_metadata": list[dict]  # RadarMetadata fields per frame
        }
    }

Module 2 is designed for the PhotonShield multi-sensor context.
The Module 1 "radar" tensor carries raw sensor data.  Since Module 1 was
designed as a generic loader (not tied to named channels), Module 2 accepts
BOTH:

A) Module 1 output directly via `from_module1_sample()` — the raw tensor is
   treated as a generic multi-channel signal where channel mapping is provided
   by the user via a channel_names list.

B) A sensor dict directly — the preferred interface for PhotonShield use:
   {
       "timestamps": {"photodiode_1": np.ndarray, ...},  # per-channel timestamps
       "values":     {"photodiode_1": np.ndarray, ...},  # per-channel raw values
   }

This design avoids any modification to Module 1 while cleanly supporting the
multi-sensor stream use case.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from module_02_sensor_dsp.config import SensorDSPConfig, ChannelConfig, MissingDataConfig
from module_02_sensor_dsp.filters import apply_filter
from module_02_sensor_dsp.denoising import apply_baseline_correction
from module_02_sensor_dsp.outliers import detect_outliers
from module_02_sensor_dsp.normalization import SensorScaler
from module_02_sensor_dsp.synchronization import synchronize_all_channels_offline
from module_02_sensor_dsp.quality import compute_all_quality


# ---------------------------------------------------------------------------
# Raw sensor data type alias
# ---------------------------------------------------------------------------
RawSensorData = Dict[str, Any]
"""Expected structure for direct sensor input:
{
    "timestamps": {channel_name: np.ndarray},
    "values":     {channel_name: np.ndarray},
}
"""

ProcessedOutput = Dict[str, Any]
"""Output structure:
{
    "signals":                {channel_name: np.ndarray (float64)},
    "timestamps":             np.ndarray (float64, unified grid),
    "validity": {
        "outlier_masks":      {channel_name: bool np.ndarray},
        "missing_masks":      {channel_name: bool np.ndarray},
        "interpolated_masks": {channel_name: bool np.ndarray},
    },
    "quality":                {channel_name: dict of quality metrics},
    "preprocessing_metadata": {
        "channels_processed": list[str],
        "sync_rate_hz": float,
        "mode": "offline" | "streaming",
        "normalization_states": {channel_name: dict},
    }
}
"""


def _handle_missing(
    values: np.ndarray,
    config: MissingDataConfig,
    causal: bool = False,
) -> np.ndarray:
    """Apply missing-data strategy. NaN tracking is done in masks; this only
    fills the values array for downstream processing.
    """
    if config.strategy == "leave":
        return values.copy()

    out = values.copy()
    if config.strategy == "constant":
        out[np.isnan(out)] = config.fill_value
        return out

    if config.strategy == "ffill":
        last = np.nan
        for i in range(len(out)):
            if np.isnan(out[i]):
                out[i] = last  # last stays nan if no prior value
            else:
                last = out[i]
        return out

    if config.strategy == "interpolate":
        if causal:
            # Streaming: fall back to forward-fill (no future data)
            return _handle_missing(values, MissingDataConfig(strategy="ffill"))
        # Offline: linear interpolation
        nan_mask = np.isnan(out)
        if not nan_mask.any():
            return out
        x = np.arange(len(out))
        valid = ~nan_mask
        out[nan_mask] = np.interp(x[nan_mask], x[valid], out[valid])
        return out

    return out


class SensorDSPPipeline:
    """Sensor Signal Preprocessing Pipeline.

    Orchestrates synchronization, filtering, baseline correction,
    outlier detection, missing-data handling, normalization, and
    signal quality computation.

    Args:
        config: SensorDSPConfig controlling all preprocessing steps.
    """

    def __init__(self, config: Optional[SensorDSPConfig] = None):
        self.config = config or SensorDSPConfig()
        # Scaler registry: channel_name -> SensorScaler
        self._scalers: Dict[str, SensorScaler] = {}
        self._scalers_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API — Offline
    # ------------------------------------------------------------------

    def fit_scalers(self, raw_data: RawSensorData) -> "SensorDSPPipeline":
        """Fit normalization scalers using TRAINING data only.

        Must be called once on training data before calling process_offline()
        on validation or test data.

        Args:
            raw_data: Dict with "values" mapping channel -> array.

        Returns:
            self (for chaining)
        """
        for ch_name, vals in raw_data.get("values", {}).items():
            ch_cfg = self.config.channels.get(ch_name)
            if ch_cfg is None:
                continue
            norm_cfg = ch_cfg.normalization
            if norm_cfg.method == "none":
                continue
            scaler = SensorScaler(
                method=norm_cfg.method,
                feature_range=tuple(norm_cfg.feature_range),
                clip=norm_cfg.clip,
            )
            sig = np.asarray(vals, dtype=np.float64)
            if not np.all(np.isnan(sig)):
                scaler.fit(sig)
                self._scalers[ch_name] = scaler

            # Optionally save state
            if norm_cfg.stats_path:
                scaler.save_state(norm_cfg.stats_path)

        self._scalers_fitted = True
        return self

    def process_offline(self, raw_data: RawSensorData) -> ProcessedOutput:
        """Process a complete sensor dataset in offline (batch) mode.

        Args:
            raw_data: Dict with:
                "timestamps": {channel_name: np.ndarray of timestamps (seconds)}
                "values":     {channel_name: np.ndarray of raw values}

        Returns:
            ProcessedOutput dict.
        """
        causal = False
        return self._process(raw_data, causal=causal)

    # ------------------------------------------------------------------
    # Public API — Streaming
    # ------------------------------------------------------------------

    def make_stream_state(self) -> Dict:
        """Create initial streaming state (one per channel)."""
        state: Dict = {
            "sync_states": {},   # channel -> {"last_time": ..., "last_value": ...}
            "filter_states": {}, # channel -> filter carry-over values
            "baseline_states": {},  # channel -> EMA baseline carry-over
        }
        for ch in self.config.channels:
            state["sync_states"][ch] = {"last_time": None, "last_value": None}
            state["filter_states"][ch] = {"ema": None, "buffer": []}
            state["baseline_states"][ch] = {"ema": None}
        return state

    def process_stream(
        self,
        sample: Dict[str, Any],
        state: Dict,
        tgt_time: float,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Process one streaming sample (causal / no future lookahead).

        Args:
            sample: Dict {channel_name: value} at the current moment.
                    Use np.nan for absent channels.
            state: Mutable streaming state from make_stream_state().
            tgt_time: Target grid time for this output sample.

        Returns:
            Tuple of (output_dict, updated_state).
            output_dict has keys: "signals", "timestamps", "validity",
            "quality" (single-sample metrics), "preprocessing_metadata".
        """
        signals: Dict[str, float] = {}
        outlier_flags: Dict[str, bool] = {}
        missing_flags: Dict[str, bool] = {}
        interp_flags: Dict[str, bool] = {}

        for ch_name, ch_cfg in self.config.channels.items():
            raw_val = sample.get(ch_name, np.nan)
            src_time = sample.get("_timestamp", tgt_time)

            # --- Streaming sync (forward-fill) ---
            sync_state = state["sync_states"].setdefault(ch_name, {"last_time": None, "last_value": None})
            from module_02_sensor_dsp.synchronization import synchronize_channel_streaming
            val, was_interp, is_missing = synchronize_channel_streaming(
                src_time=src_time,
                src_value=float(raw_val),
                tgt_time=tgt_time,
                state=sync_state,
                method=self.config.sync.method_streaming,
                max_gap_s=self.config.sync.max_gap_s,
            )
            missing_flags[ch_name] = is_missing
            interp_flags[ch_name] = was_interp

            if is_missing:
                signals[ch_name] = np.nan
                outlier_flags[ch_name] = False
                continue

            arr = np.array([val], dtype=np.float64)

            # --- EMA filter (causal) ---
            filt_state = state["filter_states"].setdefault(ch_name, {"ema": None, "buffer": []})
            filt_cfg = ch_cfg.filter
            if filt_cfg.filter_type == "ema":
                ema_val = filt_state.get("ema") or val
                new_ema = filt_cfg.alpha * val + (1.0 - filt_cfg.alpha) * ema_val
                filt_state["ema"] = new_ema
                arr = np.array([new_ema], dtype=np.float64)
            elif filt_cfg.filter_type in ("moving_average", "median"):
                buf = filt_state.setdefault("buffer", [])
                buf.append(val)
                if len(buf) > filt_cfg.window:
                    buf.pop(0)
                if filt_cfg.filter_type == "moving_average":
                    arr = np.array([np.mean(buf)], dtype=np.float64)
                else:
                    arr = np.array([np.median(buf)], dtype=np.float64)
            # Note: streaming lowpass is handled as EMA fallback (true IIR would require state)

            # --- Baseline (EMA, causal) ---
            if ch_cfg.baseline.enabled:
                b_state = state["baseline_states"].setdefault(ch_name, {"ema": None})
                b_ema = b_state.get("ema") or float(arr[0])
                new_b_ema = ch_cfg.baseline.alpha * float(arr[0]) + (1.0 - ch_cfg.baseline.alpha) * b_ema
                b_state["ema"] = new_b_ema
                arr = arr - new_b_ema

            # --- Outlier detection ---
            _, o_mask = detect_outliers(arr, ch_cfg.outlier)
            outlier_flags[ch_name] = bool(o_mask[0])

            # --- Normalization ---
            scaler = self._scalers.get(ch_name)
            if scaler and scaler._fitted:
                arr = scaler.transform(arr)

            signals[ch_name] = float(arr[0])

        return (
            {
                "signals": signals,
                "timestamps": tgt_time,
                "validity": {
                    "outlier_flags": outlier_flags,
                    "missing_flags": missing_flags,
                    "interpolated_flags": interp_flags,
                },
                "quality": None,  # single-sample quality N/A
                "preprocessing_metadata": {
                    "mode": "streaming",
                    "channels_processed": list(signals.keys()),
                    "sync_rate_hz": self.config.sync.target_rate_hz,
                },
            },
            state,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process(self, raw_data: RawSensorData, causal: bool) -> ProcessedOutput:
        """Internal offline/online processing implementation."""
        ch_times = raw_data.get("timestamps", {})
        ch_vals = raw_data.get("values", {})
        precision = np.float64 if self.config.precision == "float64" else np.float32

        # --- Synchronize all channels onto uniform grid ---
        tgt_times, synced_vals, interp_masks, missing_masks = synchronize_all_channels_offline(
            channel_times={k: np.asarray(v, dtype=np.float64) for k, v in ch_times.items()},
            channel_values={k: np.asarray(v, dtype=precision) for k, v in ch_vals.items()},
            config=self.config.sync,
        )

        n = len(tgt_times)
        processed_signals: Dict[str, np.ndarray] = {}
        outlier_masks: Dict[str, np.ndarray] = {}
        all_interp = dict(interp_masks)
        all_missing = dict(missing_masks)

        for ch_name, vals in synced_vals.items():
            ch_cfg = self.config.channels.get(ch_name)
            if ch_cfg is None:
                # Unknown channel — pass through unchanged
                processed_signals[ch_name] = vals
                outlier_masks[ch_name] = np.zeros(n, dtype=bool)
                continue

            sig = vals.astype(np.float64)

            # --- Missing data handling ---
            sig = _handle_missing(sig, ch_cfg.missing, causal=causal)

            # --- Baseline correction ---
            sig, _ = apply_baseline_correction(sig, ch_cfg.baseline, causal=causal)

            # --- Filter ---
            sig = apply_filter(sig, ch_cfg.filter, causal=causal)

            # --- Outlier detection ---
            sig, o_mask = detect_outliers(sig, ch_cfg.outlier)
            outlier_masks[ch_name] = o_mask

            # --- Normalization ---
            scaler = self._scalers.get(ch_name)
            if scaler and scaler._fitted:
                sig = scaler.transform(sig)
            elif ch_cfg.normalization.method != "none":
                # Auto-fit if no explicit scaler (e.g. offline without prior fit)
                s = SensorScaler(
                    method=ch_cfg.normalization.method,
                    feature_range=tuple(ch_cfg.normalization.feature_range),
                    clip=ch_cfg.normalization.clip,
                )
                if not np.all(np.isnan(sig)):
                    s.fit(sig)
                    self._scalers[ch_name] = s
                    sig = s.transform(sig)

            processed_signals[ch_name] = sig.astype(precision)

        # --- Signal quality ---
        quality = compute_all_quality(
            channel_values=processed_signals,
            outlier_masks=outlier_masks,
            missing_masks=all_missing,
            interpolated_masks=all_interp,
            timestamps=tgt_times,
        )

        norm_states = {
            ch: s.get_state() for ch, s in self._scalers.items()
        }

        return {
            "signals": processed_signals,
            "timestamps": tgt_times,
            "validity": {
                "outlier_masks": outlier_masks,
                "missing_masks": all_missing,
                "interpolated_masks": all_interp,
            },
            "quality": quality,
            "preprocessing_metadata": {
                "channels_processed": list(processed_signals.keys()),
                "sync_rate_hz": self.config.sync.target_rate_hz,
                "mode": "offline",
                "normalization_states": norm_states,
            },
        }

    # ------------------------------------------------------------------
    # Module 1 compatibility bridge
    # ------------------------------------------------------------------

    @staticmethod
    def from_module1_sample(
        sample: Dict[str, Any],
        channel_names: Optional[List[str]] = None,
    ) -> RawSensorData:
        """Convert a Module 1 output sample to Module 2 raw_data format.

        Module 1 produces:
            sample["radar"]     -> torch.Tensor shape [T, C] or [T, ...]
            sample["timestamp"] -> torch.Tensor shape [T]

        This bridge treats each column of the radar tensor as a separate channel.

        Args:
            sample: Module 1 `RadarDataset.__getitem__` output.
            channel_names: Optional list of channel names for each column in
                           the radar tensor. If None, channels are named
                           "channel_0", "channel_1", etc.

        Returns:
            RawSensorData dict compatible with process_offline() / process_stream().

        Notes:
            - If the radar tensor has more than 2 dimensions, it is flattened
              along the feature axis: shape [T, *] -> [T, flat_features].
            - Timestamps from Module 1 are used as-is (float64 seconds).
        """
        import torch

        radar: "torch.Tensor" = sample["radar"]   # [T, ...]
        timestamps: "torch.Tensor" = sample["timestamp"]  # [T]

        radar_np = radar.numpy().astype(np.float64)
        ts_np = timestamps.numpy().astype(np.float64)

        # Flatten all dims after T
        T = radar_np.shape[0]
        flat = radar_np.reshape(T, -1)  # [T, F]
        n_channels = flat.shape[1]

        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(n_channels)]
        elif len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names length {len(channel_names)} != "
                f"number of features {n_channels}"
            )

        times_dict = {name: ts_np for name in channel_names}
        vals_dict = {name: flat[:, i] for i, name in enumerate(channel_names)}

        return {"timestamps": times_dict, "values": vals_dict}
