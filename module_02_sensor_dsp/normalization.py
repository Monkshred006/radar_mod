"""Normalization and scaling utilities with training-statistics preservation.

Prevents data leakage by separating fit() and transform() steps:
- fit() computes statistics from TRAINING data only.
- transform() applies scaling using previously fitted statistics.
- Statistics are serializable to/from JSON for inference reproducibility.
- inverse_transform() recovers original scale where appropriate.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np

from module_02_sensor_dsp.config import NormalizationConfig


class SensorScaler:
    """Normalization scaler for a single sensor channel.

    Supports: "minmax", "standard", "robust", "none".

    Usage (training):
        scaler = SensorScaler("minmax")
        scaler.fit(train_signal)
        scaled = scaler.transform(train_signal)
        scaler.save_state("stats/photodiode_1.json")

    Usage (inference/validation):
        scaler = SensorScaler.load_state("stats/photodiode_1.json")
        scaled = scaler.transform(val_signal)
    """

    def __init__(self, method: str = "none", feature_range: tuple = (0.0, 1.0), clip: bool = False):
        self.method = method
        self.feature_range = feature_range
        self.clip = clip
        self._fitted = False
        # Statistics
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._median: Optional[float] = None
        self._iqr: Optional[float] = None

    def fit(self, signal: np.ndarray) -> "SensorScaler":
        """Compute normalization statistics from signal.

        Args:
            signal: 1-D float array (NaNs excluded from statistics computation).

        Returns:
            self (for chaining)
        """
        valid = signal[~np.isnan(signal)].astype(np.float64)
        if len(valid) == 0:
            raise ValueError("Cannot fit scaler on all-NaN signal.")

        self._mean = float(np.mean(valid))
        self._std = float(np.std(valid)) or 1.0  # guard division by zero
        self._min = float(np.min(valid))
        self._max = float(np.max(valid))
        self._median = float(np.median(valid))
        q1 = float(np.percentile(valid, 25))
        q3 = float(np.percentile(valid, 75))
        self._iqr = (q3 - q1) or 1.0  # guard division by zero
        self._fitted = True
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply normalization using fitted statistics.

        NaN values are preserved (not altered by normalization).

        Args:
            signal: 1-D float array.

        Returns:
            Normalized signal (float64), NaNs preserved.
        """
        if not self._fitted:
            raise RuntimeError("SensorScaler must be fit() before transform().")
        if self.method == "none":
            return signal.astype(np.float64)

        sig = signal.astype(np.float64)
        nan_mask = np.isnan(sig)
        out = sig.copy()

        if self.method == "minmax":
            rng = self._max - self._min
            if rng == 0:
                out[~nan_mask] = self.feature_range[0]
            else:
                lo, hi = self.feature_range
                out[~nan_mask] = lo + (sig[~nan_mask] - self._min) / rng * (hi - lo)
            if self.clip:
                out[~nan_mask] = np.clip(out[~nan_mask], *self.feature_range)

        elif self.method == "standard":
            out[~nan_mask] = (sig[~nan_mask] - self._mean) / self._std

        elif self.method == "robust":
            out[~nan_mask] = (sig[~nan_mask] - self._median) / self._iqr

        else:
            raise ValueError(f"Unknown normalization method: {self.method!r}")

        out[nan_mask] = np.nan
        return out

    def inverse_transform(self, signal: np.ndarray) -> np.ndarray:
        """Recover original scale from normalized values.

        Args:
            signal: Normalized 1-D float array.

        Returns:
            De-normalized signal.
        """
        if not self._fitted:
            raise RuntimeError("SensorScaler must be fit() before inverse_transform().")
        if self.method == "none":
            return signal.astype(np.float64)

        sig = signal.astype(np.float64)
        nan_mask = np.isnan(sig)
        out = sig.copy()

        if self.method == "minmax":
            lo, hi = self.feature_range
            rng = self._max - self._min
            out[~nan_mask] = (sig[~nan_mask] - lo) / (hi - lo) * rng + self._min

        elif self.method == "standard":
            out[~nan_mask] = sig[~nan_mask] * self._std + self._mean

        elif self.method == "robust":
            out[~nan_mask] = sig[~nan_mask] * self._iqr + self._median

        out[nan_mask] = np.nan
        return out

    def get_state(self) -> Dict[str, Any]:
        """Return serializable state dict."""
        return {
            "method": self.method,
            "feature_range": list(self.feature_range),
            "clip": self.clip,
            "mean": self._mean,
            "std": self._std,
            "min": self._min,
            "max": self._max,
            "median": self._median,
            "iqr": self._iqr,
            "fitted": self._fitted,
        }

    def save_state(self, filepath: Union[str, Path]) -> None:
        """Serialize statistics to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.get_state(), f, indent=2)

    @classmethod
    def load_state(cls, filepath: Union[str, Path]) -> "SensorScaler":
        """Load a previously fitted scaler from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        scaler = cls(
            method=state["method"],
            feature_range=tuple(state["feature_range"]),
            clip=state.get("clip", False),
        )
        scaler._mean = state.get("mean")
        scaler._std = state.get("std")
        scaler._min = state.get("min")
        scaler._max = state.get("max")
        scaler._median = state.get("median")
        scaler._iqr = state.get("iqr")
        scaler._fitted = state.get("fitted", False)
        return scaler

    @classmethod
    def from_config(cls, config: NormalizationConfig) -> "SensorScaler":
        """Build a SensorScaler from NormalizationConfig.

        If config.stats_path exists, loads pre-fitted statistics.
        """
        scaler = cls(
            method=config.method,
            feature_range=tuple(config.feature_range),
            clip=config.clip,
        )
        if config.stats_path and Path(config.stats_path).exists():
            loaded = cls.load_state(config.stats_path)
            scaler._mean = loaded._mean
            scaler._std = loaded._std
            scaler._min = loaded._min
            scaler._max = loaded._max
            scaler._median = loaded._median
            scaler._iqr = loaded._iqr
            scaler._fitted = loaded._fitted
        return scaler
