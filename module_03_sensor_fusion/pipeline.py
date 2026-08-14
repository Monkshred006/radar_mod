"""Main Sensor Fusion Pipeline for Module 3 — Sensor Fusion + Feature Extraction.

Exposes:
- process_offline(module2_output) -> FusionOutput
- process_stream(module2_stream_sample, state) -> (FusionOutput, state)

Output dict structure:
{
    "features":      torch.Tensor [B, T, F_total] or [T, F_total],
    "tokens":        torch.Tensor [B, T, S, D_features] or [T, S, D_features],
    "token_mask":    torch.Tensor [B, T, S, D_features],
    "timestamps":    torch.Tensor [T],
    "sensor_groups": list[str],
    "feature_names": list[str],
    "group_map":     dict,
    "quality":       dict from Module 2,
    "metadata":      dict,
}
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch

from module_03_sensor_fusion.config import Module3Config
from module_03_sensor_fusion.sensor_groups import SensorGroupRegistry
from module_03_sensor_fusion.fusion import fuse_sensor_features
from module_03_sensor_fusion.tokenization import SensorAwareTokenizer

FusionOutput = Dict[str, Any]


class SensorFusionPipeline:
    """Sensor Fusion & Feature Extraction Pipeline.

    Consumes Module 2 ProcessedOutput and extracts sensor-specific features,
    fuses them chronologically, and packages them into sensor-aware 4D tokens.
    """

    def __init__(self, config: Optional[Module3Config] = None):
        self.config = config or Module3Config()
        self.registry = SensorGroupRegistry()
        self.tokenizer = SensorAwareTokenizer(self.config.tokenizer)

    def process_offline(self, module2_output: Dict[str, Any]) -> FusionOutput:
        """Process a complete dataset or sequence in offline mode.

        Args:
            module2_output: Dict returned by Module 2 `process_offline()`:
                {
                    "signals": {channel_name: np.ndarray [T]},
                    "timestamps": np.ndarray [T],
                    "validity": {outlier_masks, missing_masks, interpolated_masks},
                    "quality": dict,
                    "preprocessing_metadata": dict,
                }

        Returns:
            FusionOutput dictionary.
        """
        signals = module2_output["signals"]
        timestamps = module2_output["timestamps"]
        validity = module2_output.get("validity", {})
        quality = module2_output.get("quality", {})

        target_dtype = torch.float64 if self.config.dtype == "float64" else torch.float32

        # In streaming mode offline processing also uses causal features
        # so offline and streaming modes stay equivalent under causal config.
        causal = self.config.streaming
        fused_np, feat_names, group_map = fuse_sensor_features(
            signals=signals,
            timestamps=timestamps,
            validity_dict=validity,
            config=self.config,
            registry=self.registry,
            causal=causal,
        )

        # Build tokens [T, S, D_features]
        tokens_tensor, mask_tensor, active_groups = self.tokenizer.build_tokens_single(
            fused_np, group_map, dtype=target_dtype
        )

        features_tensor = torch.from_numpy(fused_np).to(dtype=target_dtype)
        ts_tensor = torch.from_numpy(timestamps).to(dtype=torch.float64)

        return {
            "features": features_tensor,                # [T, F_fused]
            "tokens": tokens_tensor,                    # [T, S, D_max]
            "token_mask": mask_tensor,                  # [T, S, D_max]
            "timestamps": ts_tensor,                    # [T]
            "sensor_groups": active_groups,
            "feature_names": feat_names,
            "group_map": group_map,
            "quality": quality,
            "metadata": {
                "mode": "offline",
                "n_timesteps": len(timestamps),
                "n_features": len(feat_names),
                "n_groups": len(active_groups),
            },
        }

    def make_stream_state(self) -> Dict[str, Any]:
        """Initialize streaming state for Module 3."""
        return {
            "history_signals": {},      # channel -> list of past values
            "history_timestamps": [],   # list of past timestamps
            "validity_history": {
                "outlier_masks": {},
                "missing_masks": {},
                "interpolated_masks": {},
            },
        }

    def process_stream(
        self,
        module2_stream_output: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Tuple[FusionOutput, Dict[str, Any]]:
        """Process a single streaming sample in causal mode.

        Args:
            module2_stream_output: Output from Module 2 `process_stream()`.
            state: Streaming state dict.

        Returns:
            Tuple of (FusionOutput for current sample, updated state).
        """
        target_dtype = torch.float64 if self.config.dtype == "float64" else torch.float32

        sig_val = module2_stream_output["signals"]          # {ch: float}
        ts_val = module2_stream_output["timestamps"]         # float
        val_flags = module2_stream_output["validity"]        # {outlier_flags, missing_flags, interpolated_flags}

        # Update history arrays
        state["history_timestamps"].append(ts_val)
        ts_hist = np.array(state["history_timestamps"], dtype=np.float64)

        sig_hist: Dict[str, np.ndarray] = {}
        for ch, v in sig_val.items():
            h = state["history_signals"].setdefault(ch, [])
            h.append(v)
            sig_hist[ch] = np.array(h, dtype=np.float64)

        val_dict_hist: Dict[str, Dict[str, np.ndarray]] = {
            "outlier_masks": {},
            "missing_masks": {},
            "interpolated_masks": {},
        }
        for ch in sig_val:
            o_h = state["validity_history"]["outlier_masks"].setdefault(ch, [])
            m_h = state["validity_history"]["missing_masks"].setdefault(ch, [])
            i_h = state["validity_history"]["interpolated_masks"].setdefault(ch, [])

            o_h.append(val_flags.get("outlier_flags", {}).get(ch, False))
            m_h.append(val_flags.get("missing_flags", {}).get(ch, False))
            i_h.append(val_flags.get("interpolated_flags", {}).get(ch, False))

            val_dict_hist["outlier_masks"][ch] = np.array(o_h, dtype=bool)
            val_dict_hist["missing_masks"][ch] = np.array(m_h, dtype=bool)
            val_dict_hist["interpolated_masks"][ch] = np.array(i_h, dtype=bool)

        # Fuse features causally across history
        fused_hist, feat_names, group_map = fuse_sensor_features(
            signals=sig_hist,
            timestamps=ts_hist,
            validity_dict=val_dict_hist,
            config=self.config,
            registry=self.registry,
            causal=True,
        )

        # Extract latest timestep features for streaming output
        fused_latest = fused_hist[-1:, :]  # [1, F_total]
        tokens_latest, mask_latest, active_groups = self.tokenizer.build_tokens_single(
            fused_latest, group_map, dtype=target_dtype
        )

        features_tensor = torch.from_numpy(fused_latest).to(dtype=target_dtype)
        ts_tensor = torch.tensor([ts_val], dtype=torch.float64)

        return {
            "features": features_tensor,          # [1, F_fused]
            "tokens": tokens_latest,              # [1, S, D_max]
            "token_mask": mask_latest,            # [1, S, D_max]
            "timestamps": ts_tensor,              # [1]
            "sensor_groups": active_groups,
            "feature_names": feat_names,
            "group_map": group_map,
            "quality": module2_stream_output.get("quality"),
            "metadata": {
                "mode": "streaming",
                "n_history_samples": len(ts_hist),
            },
        }, state
