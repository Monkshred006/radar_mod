"""Early stopping utility for Module 5."""

from __future__ import annotations
from typing import Literal, Optional


class EarlyStopping:
    """Generic configurable early stopping.

    Monitors a validation metric and signals the training loop to stop
    when no improvement is observed for `patience` consecutive epochs.

    Args:
        monitor: Name of the metric to watch (e.g. "val_loss", "val_f1_macro").
        mode: "min" if lower is better (loss), "max" if higher is better (F1).
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        patience: int = 10,
        min_delta: float = 1e-5,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self._best: Optional[float] = None
        self._counter: int = 0
        self.stopped_epoch: Optional[int] = None

    def _is_improvement(self, current: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return current < self._best - self.min_delta
        else:
            return current > self._best + self.min_delta

    def step(self, metric_value: float, epoch: int) -> bool:
        """Update state with the latest metric value.

        Args:
            metric_value: Current epoch's monitored metric.
            epoch: Current epoch number (1-indexed).

        Returns:
            True if training should stop, False otherwise.
        """
        if self._is_improvement(metric_value):
            self._best = metric_value
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    @property
    def best(self) -> Optional[float]:
        return self._best

    @property
    def wait_count(self) -> int:
        return self._counter

    def state_dict(self) -> dict:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self._best,
            "counter": self._counter,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        self._best = state.get("best")
        self._counter = state.get("counter", 0)
        self.stopped_epoch = state.get("stopped_epoch")
