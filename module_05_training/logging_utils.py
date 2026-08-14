"""Lightweight experiment logger for Module 5.

Records per-epoch metrics to JSON and CSV without requiring any
external experiment tracking service.
"""

from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Records per-epoch training metrics to JSON and CSV.

    Args:
        log_dir: Directory where log files will be written.
        experiment_name: Name prefix for log files.
    """

    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.json_path = self.log_dir / f"{experiment_name}_log.json"
        self.csv_path = self.log_dir / f"{experiment_name}_log.csv"
        self.history: List[Dict[str, Any]] = []
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None
        self._csv_headers_written = False

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        metrics: Dict[str, float],
        epoch_duration_s: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one epoch of metrics.

        Args:
            epoch: 1-indexed epoch number.
            train_loss: Average training loss for this epoch.
            val_loss: Average validation loss for this epoch.
            learning_rate: Current learning rate.
            metrics: Dict of metric_name → float.
            epoch_duration_s: Wall-clock time for this epoch in seconds.
            extra: Optional extra key-value pairs to include.
        """
        record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "epoch_duration_s": epoch_duration_s,
            **metrics,
        }
        if extra:
            record.update(extra)
        self.history.append(record)

        # JSON: overwrite with full history each time (safe for crash recovery)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, default=str)

        # CSV: append row
        self._write_csv_row(record)

    def _write_csv_row(self, record: Dict[str, Any]) -> None:
        """Append one row to the CSV log."""
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the experiment history."""
        if not self.history:
            return {}
        best = min(self.history, key=lambda r: r.get("val_loss", float("inf")))
        return {
            "total_epochs": len(self.history),
            "best_epoch": best["epoch"],
            "best_val_loss": best.get("val_loss"),
            "final_train_loss": self.history[-1].get("train_loss"),
            "total_training_time_s": sum(r.get("epoch_duration_s", 0) for r in self.history),
            "json_log": str(self.json_path),
            "csv_log": str(self.csv_path),
        }
