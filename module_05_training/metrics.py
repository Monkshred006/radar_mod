"""Metrics framework for Module 5.

Accumulates predictions/targets across batches and computes per-epoch metrics.
Only computes metrics appropriate for the configured task type.
Does NOT fabricate metric values.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import torch
import numpy as np


class MetricsTracker:
    """Accumulates model outputs and targets across batches.

    Call update() per batch, then compute() at epoch end.
    Reset with reset() between epochs.
    """

    def __init__(self, task_type: str = "regression"):
        """
        Args:
            task_type: "regression", "classification", or "multitask".
        """
        self.task_type = task_type
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def reset(self) -> None:
        """Clear accumulated state for a new epoch."""
        self._preds.clear()
        self._targets.clear()

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch of predictions and targets.

        Args:
            predictions: Model outputs [B, ...] (raw logits or values).
            targets: Ground-truth targets [B, ...].
        """
        self._preds.append(predictions.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated data.

        Returns:
            Dict of metric_name → float value.
        """
        if not self._preds:
            return {}

        preds = torch.cat(self._preds, dim=0)    # [N, ...]
        targets = torch.cat(self._targets, dim=0)  # [N, ...]

        if self.task_type == "classification":
            return self._classification_metrics(preds, targets)
        elif self.task_type == "regression":
            return self._regression_metrics(preds, targets)
        else:
            return {}

    # ── Classification ────────────────────────────────────────────────────────

    def _classification_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Accuracy, precision, recall, F1 (macro)."""
        if logits.ndim == 1 or logits.shape[-1] == 1:
            # Binary: sigmoid threshold at 0.5
            probs = torch.sigmoid(logits.squeeze(-1))
            pred_labels = (probs >= 0.5).long()
            targets = targets.squeeze(-1).long()
            num_classes = 2
        else:
            pred_labels = logits.argmax(dim=-1)
            targets = targets.squeeze(-1).long()
            num_classes = logits.shape[-1]

        n = targets.shape[0]
        correct = (pred_labels == targets).sum().item()
        accuracy = correct / n if n > 0 else 0.0

        # Per-class precision, recall, F1
        precision_list, recall_list, f1_list = [], [], []
        for c in range(num_classes):
            tp = ((pred_labels == c) & (targets == c)).sum().item()
            fp = ((pred_labels == c) & (targets != c)).sum().item()
            fn = ((pred_labels != c) & (targets == c)).sum().item()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precision_list.append(p)
            recall_list.append(r)
            f1_list.append(f)

        return {
            "accuracy": accuracy,
            "precision_macro": float(np.mean(precision_list)),
            "recall_macro": float(np.mean(recall_list)),
            "f1_macro": float(np.mean(f1_list)),
        }

    # ── Regression ────────────────────────────────────────────────────────────

    def _regression_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """MAE, MSE, RMSE, R²."""
        preds = preds.float().reshape(-1)
        targets = targets.float().reshape(-1)

        n = preds.shape[0]
        if n == 0:
            return {}

        diff = preds - targets
        mae = diff.abs().mean().item()
        mse = (diff ** 2).mean().item()
        rmse = math.sqrt(mse)

        # R² = 1 - SS_res / SS_tot
        ss_res = (diff ** 2).sum().item()
        ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }


class MultiTaskMetricsTracker:
    """Per-task MetricsTracker for multi-task learning."""

    def __init__(self, task_configs: Dict[str, str]):
        """
        Args:
            task_configs: {task_name: task_type} e.g.
                {"cls": "classification", "reg": "regression"}
        """
        self.trackers: Dict[str, MetricsTracker] = {
            name: MetricsTracker(task_type=ttype)
            for name, ttype in task_configs.items()
        }

    def reset(self) -> None:
        for t in self.trackers.values():
            t.reset()

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> None:
        for name, tracker in self.trackers.items():
            if name in predictions and name in targets:
                tracker.update(predictions[name], targets[name])

    def compute(self) -> Dict[str, float]:
        """Returns flat dict: {task_name/metric_name: value}."""
        results: Dict[str, float] = {}
        for name, tracker in self.trackers.items():
            task_metrics = tracker.compute()
            for k, v in task_metrics.items():
                results[f"{name}/{k}"] = v
        return results
