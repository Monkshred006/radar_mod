"""Evaluator for Module 5 — Test-Set Evaluation.

IMPORTANT: The test set must be evaluated ONLY after model selection
is complete (i.e., after training + validation are done).
Do NOT use test metrics for early stopping or hyperparameter tuning.
"""

from __future__ import annotations
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module_05_training.config import TrainingConfig
from module_05_training.losses import get_loss_fn
from module_05_training.metrics import MetricsTracker
from module_05_training.trainer import _move_dict_to_device, _move_to_device


class Evaluator:
    """Evaluates a trained model on the test set.

    Args:
        config: TrainingConfig specifying loss, metrics, output_key, device.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if (config.device == "auto" and torch.cuda.is_available())
            else config.device if config.device != "auto" else "cpu"
        )
        self.loss_fn = get_loss_fn(config)
        self.metrics_tracker = MetricsTracker(task_type=config.target_type)

    @torch.no_grad()
    def evaluate(
        self,
        engine: nn.Module,
        head: nn.Module,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Evaluate engine + head on the test DataLoader.

        Args:
            engine: Trained Module 4 engine (PhotonMambaHybrid).
            head: Trained task head.
            test_loader: DataLoader over the held-out test set.

        Returns:
            Dict containing:
                - loss: float
                - metrics: dict of metric_name → float
                - inference_time_s: total inference wall-clock time
                - per_sample_latency_ms: average ms per sample
                - sample_count: number of samples evaluated
        """
        engine = engine.to(self.device)
        head = head.to(self.device)
        engine.eval()
        head.eval()
        self.metrics_tracker.reset()

        total_loss = 0.0
        n_batches = 0
        sample_count = 0

        t0 = time.time()
        for module3_dict, targets in test_loader:
            module3_dict = _move_dict_to_device(module3_dict, self.device)
            targets = _move_to_device(targets, self.device)

            engine_out = engine(module3_dict)
            selected = engine_out[self.config.output_key]
            prediction = head(selected)

            loss = self.loss_fn(prediction, targets)
            total_loss += loss.item()
            n_batches += 1
            sample_count += targets.shape[0] if isinstance(targets, torch.Tensor) else 1
            self.metrics_tracker.update(prediction, targets)

        inference_time_s = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        metrics = self.metrics_tracker.compute()
        per_sample_latency_ms = (
            (inference_time_s * 1000 / sample_count) if sample_count > 0 else float("nan")
        )

        return {
            "loss": avg_loss,
            "metrics": metrics,
            "inference_time_s": inference_time_s,
            "per_sample_latency_ms": per_sample_latency_ms,
            "sample_count": sample_count,
        }
