"""Generic Target Adapter for Module 5.

Converts dataset samples to training targets without assuming
a specific final task representation.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import numpy as np

from module_05_training.config import TrainingConfig


class TargetAdapter:
    """Configurable target extractor for training samples.

    Reads a target from a dataset sample dict using a configurable key
    or callable, converts it to an appropriate torch.Tensor dtype,
    and validates the shape.

    Args:
        target_fn: Callable(sample_dict) -> target_tensor OR a string key
            that is looked up in the sample dict.
        target_type: "regression", "classification", or "multitask".
        num_classes: Number of classes for classification targets.
        num_outputs: Number of outputs for regression targets.
        dtype: Target tensor dtype. Inferred from target_type if None.
    """

    def __init__(
        self,
        target_fn: Callable[[Dict[str, Any]], Any],
        target_type: str = "regression",
        num_classes: int = 2,
        num_outputs: int = 1,
        dtype: Optional[torch.dtype] = None,
    ):
        self.target_fn = target_fn
        self.target_type = target_type
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self._dtype = dtype

    @property
    def dtype(self) -> torch.dtype:
        if self._dtype is not None:
            return self._dtype
        if self.target_type == "classification":
            return torch.long
        return torch.float32

    def __call__(self, sample: Dict[str, Any]) -> Any:
        """Extract and convert target from sample dict.

        Args:
            sample: Dataset sample dictionary.

        Returns:
            Target tensor (or dict of tensors for multitask).
        """
        raw = self.target_fn(sample)

        if self.target_type == "multitask":
            # raw expected to be a dict of {task_name: value}
            return {
                k: self._to_tensor(v) for k, v in raw.items()
            }

        return self._to_tensor(raw)

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            t = value
        elif isinstance(value, np.ndarray):
            t = torch.from_numpy(value)
        else:
            t = torch.tensor(value)

        t = t.to(dtype=self.dtype)

        if t.ndim == 0:
            t = t.unsqueeze(0)

        return t


class SyntheticRegressionAdapter(TargetAdapter):
    """Deterministic synthetic regression target for testing and sanity checks.

    Target = mean of the first channel of 'features' in the Module 3 dict.
    This is a reproducible, non-random target that allows training loss to
    decrease as a sanity check.

    IMPORTANT: This adapter produces synthetic targets only for testing.
    It does NOT represent any real PhotonShield sensing task.
    """

    def __init__(self, num_outputs: int = 1):
        def _target_fn(sample: Dict[str, Any]) -> torch.Tensor:
            features = sample.get("features")
            if features is None:
                tokens = sample.get("tokens")
                if tokens is not None:
                    # Use mean of all token features as synthetic target
                    val = tokens.to(torch.float32).mean(dim=(0, 1, 2))[:num_outputs]
                else:
                    val = torch.zeros(num_outputs, dtype=torch.float32)
            else:
                feat = features.to(torch.float32)
                # Deterministic: mean over T, then first num_outputs features
                reduced = feat.mean(dim=0)[:num_outputs]
                val = reduced

            if val.shape[0] < num_outputs:
                pad = torch.zeros(num_outputs - val.shape[0])
                val = torch.cat([val, pad])
            return val.reshape(num_outputs)

        super().__init__(
            target_fn=_target_fn,
            target_type="regression",
            num_outputs=num_outputs,
            dtype=torch.float32,
        )


class SyntheticClassificationAdapter(TargetAdapter):
    """Deterministic synthetic binary classification target for testing.

    Target = (sum of tokens > 0) → binary label 0 or 1.
    Reproducible and non-random for sanity checks only.
    """

    def __init__(self, num_classes: int = 2):
        def _target_fn(sample: Dict[str, Any]) -> torch.Tensor:
            features = sample.get("features")
            if features is None:
                tokens = sample.get("tokens")
                val = tokens.to(torch.float32).sum().item() if tokens is not None else 0.0
            else:
                val = features.to(torch.float32).sum().item()
            return torch.tensor(int(val > 0) % num_classes, dtype=torch.long)

        super().__init__(
            target_fn=_target_fn,
            target_type="classification",
            num_classes=num_classes,
            dtype=torch.long,
        )


def get_target_adapter(config: TrainingConfig) -> TargetAdapter:
    """Factory: create the appropriate adapter based on training config.

    Defaults to synthetic adapters when no real target source is configured.
    """
    if config.target_type == "classification":
        return SyntheticClassificationAdapter(num_classes=config.num_classes)
    else:
        return SyntheticRegressionAdapter(num_outputs=config.num_regression_outputs)
