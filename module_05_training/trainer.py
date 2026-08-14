"""Main Trainer for Module 5 — FP32 Training Loop.

Orchestrates:
  model.train()  →  forward  →  loss  →  backward  →  clip  →  step
  model.eval()   →  validate →  metrics  →  checkpoint  →  early_stop
"""

from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module_05_training.config import TrainingConfig
from module_05_training.losses import TrainingNaNError, get_loss_fn
from module_05_training.metrics import MetricsTracker
from module_05_training.optimizer import get_optimizer
from module_05_training.scheduler import get_scheduler
from module_05_training.early_stopping import EarlyStopping
from module_05_training.checkpointing import save_checkpoint, load_checkpoint
from module_05_training.logging_utils import ExperimentLogger
from module_05_training.reproducibility import set_seed


def _resolve_device(config: TrainingConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


class Trainer:
    """FP32 training loop for PhotonShield Mamba-Hybrid model.

    Usage::

        trainer = Trainer(engine, head, config)
        trainer.fit(train_loader, val_loader)

    Args:
        engine: PhotonMambaHybrid (or any nn.Module).
        head: Task head (ClassificationHead, RegressionHead, etc.).
        config: TrainingConfig.
        model_config: Module 4 config (saved in checkpoints).
        target_adapter: Optional callable transforming Module 3 dict → target.
            Not used during training iteration (targets come from DataLoader).
        experiment_name: Name prefix for log files.
    """

    def __init__(
        self,
        engine: nn.Module,
        head: nn.Module,
        config: TrainingConfig,
        model_config: Any = None,
        experiment_name: str = "photonshield_m5",
    ):
        config.validate()
        self.config = config
        self.model_config = model_config
        self.experiment_name = experiment_name

        self.device = _resolve_device(config)
        print(f"[Trainer] Device: {self.device}")

        # Move models to device
        self.engine = engine.to(self.device)
        self.head = head.to(self.device)

        # Combined parameter list for optimizer
        all_params = list(engine.parameters()) + list(head.parameters())

        # Build a combined module for optimizer convenience
        self._combined = nn.ModuleList([engine, head])

        self.loss_fn = get_loss_fn(config)
        self.optimizer = get_optimizer(self._combined, config)
        self.metrics_tracker = MetricsTracker(task_type=config.target_type)
        self.early_stopping = EarlyStopping(
            monitor=config.early_stopping_monitor,
            mode=config.early_stopping_mode,
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )
        self.logger = ExperimentLogger(config.log_dir, experiment_name)

        self.history: List[Dict[str, Any]] = []
        self.best_val_metric = (
            float("inf") if config.early_stopping_mode == "min" else float("-inf")
        )
        self._scheduler = None  # initialised in fit() once we know loader len

        # Resume from checkpoint if specified
        self.start_epoch = 1
        if config.resume_from and Path(config.resume_from).exists():
            self._resume(config.resume_from)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader yielding (module3_dict, target) batches.
            val_loader: DataLoader for validation.

        Returns:
            Training summary dict.
        """
        set_seed(self.config.random_seed)

        self._scheduler = get_scheduler(
            self.optimizer, self.config, len(train_loader)
        )

        print(f"[Trainer] Starting training: {self.config.epochs} epochs")
        print(f"[Trainer] Train batches/epoch: {len(train_loader)}")
        print(f"[Trainer] Val batches/epoch:   {len(val_loader)}")

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            t_start = time.time()

            # ── Train ─────────────────────────────────────────────────────────
            train_loss, grad_norm = self.train_epoch(train_loader, epoch)

            # ── Validate ──────────────────────────────────────────────────────
            val_loss, val_metrics = {}, {}
            if epoch % self.config.val_every_n_epochs == 0:
                val_loss_val, val_metrics = self.validate(val_loader)
            else:
                val_loss_val = float("nan")

            # ── LR step ───────────────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self._scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self._scheduler, ReduceLROnPlateau):
                    if not math.isnan(val_loss_val):
                        self._scheduler.step(val_loss_val)
                else:
                    self._scheduler.step()

            epoch_time = time.time() - t_start

            # ── Log ───────────────────────────────────────────────────────────
            print(
                f"  Epoch {epoch:3d}/{self.config.epochs} | "
                f"train_loss={train_loss:.5f} | val_loss={val_loss_val:.5f} | "
                f"lr={current_lr:.2e} | grad_norm={grad_norm:.3f} | "
                f"time={epoch_time:.1f}s"
            )
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss_val,
                learning_rate=current_lr,
                metrics=val_metrics,
                epoch_duration_s=epoch_time,
                extra={"grad_norm": grad_norm},
            )
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_val,
                **val_metrics,
            })

            # ── Checkpoint ────────────────────────────────────────────────────
            monitored = val_metrics.get(
                self.config.early_stopping_monitor.replace("val_", ""),
                val_loss_val,
            )
            if not math.isnan(monitored):
                is_best = self._is_better(monitored, self.best_val_metric)
                if is_best:
                    self.best_val_metric = monitored
                    if self.config.save_best:
                        self._save(epoch, suffix="best")
                if self.config.save_latest:
                    self._save(epoch, suffix="latest")

                # ── Early stopping ────────────────────────────────────────────
                if self.early_stopping.step(monitored, epoch):
                    print(
                        f"[Trainer] Early stopping at epoch {epoch}. "
                        f"Best {self.config.early_stopping_monitor}: {self.best_val_metric:.5f}"
                    )
                    break

        summary = self.logger.summary()
        summary["best_val_metric"] = self.best_val_metric
        print(f"\n[Trainer] Training complete. Best val metric: {self.best_val_metric:.5f}")
        return summary

    def train_epoch(
        self, loader: DataLoader, epoch: int = 0
    ) -> Tuple[float, float]:
        """Train for one epoch.

        Returns:
            (avg_loss, avg_grad_norm)
        """
        self.engine.train()
        self.head.train()

        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, (module3_dict, targets) in enumerate(loader):
            # Move tensors to device
            module3_dict = _move_dict_to_device(module3_dict, self.device)
            targets = _move_to_device(targets, self.device)

            # ── Forward ───────────────────────────────────────────────────────
            engine_out = self.engine(module3_dict)
            selected = engine_out[self.config.output_key]  # e.g. [B, D_model]
            prediction = self.head(selected)               # [B, num_outputs]

            # ── Loss ──────────────────────────────────────────────────────────
            loss = self.loss_fn(prediction, targets)
            loss_scaled = loss / self.config.gradient_accumulation_steps
            loss_scaled.backward()

            # ── Gradient accumulation ─────────────────────────────────────────
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # NaN/Inf gradient detection
                grad_norm = self._check_and_clip_gradients()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_grad_norm += grad_norm

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_grad_norm = total_grad_norm / max(
            n_batches // self.config.gradient_accumulation_steps, 1
        )
        return avg_loss, avg_grad_norm

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Run validation.

        Returns:
            (avg_val_loss, metrics_dict)
        """
        self.engine.eval()
        self.head.eval()
        self.metrics_tracker.reset()

        total_loss = 0.0
        n_batches = 0

        for module3_dict, targets in loader:
            module3_dict = _move_dict_to_device(module3_dict, self.device)
            targets = _move_to_device(targets, self.device)

            engine_out = self.engine(module3_dict)
            selected = engine_out[self.config.output_key]
            prediction = self.head(selected)

            loss = self.loss_fn(prediction, targets)
            total_loss += loss.item()
            n_batches += 1

            self.metrics_tracker.update(prediction, targets)

        avg_loss = total_loss / max(n_batches, 1)
        metrics = self.metrics_tracker.compute()
        return avg_loss, metrics

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save training checkpoint."""
        save_checkpoint(
            path=path,
            model=self._combined,
            optimizer=self.optimizer,
            scheduler=self._scheduler,
            epoch=epoch,
            best_val_metric=self.best_val_metric,
            training_config=self.config,
            model_config=self.model_config,
            history=self.history,
            seed=self.config.random_seed,
        )

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint into current trainer state."""
        ckpt = load_checkpoint(
            path=path,
            model=self._combined,
            optimizer=self.optimizer,
            scheduler=self._scheduler,
            device=self.device,
        )
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_val_metric = ckpt.get("best_val_metric", self.best_val_metric)
        self.history = ckpt.get("history", [])
        return ckpt

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _check_and_clip_gradients(self) -> float:
        """Check for NaN/Inf gradients, then clip. Returns grad norm."""
        params_with_grad = [
            p for p in self._combined.parameters()
            if p.requires_grad and p.grad is not None
        ]
        # NaN/Inf detection
        for p in params_with_grad:
            if not torch.isfinite(p.grad).all():
                raise TrainingNaNError(
                    "NaN or Inf detected in gradients. "
                    "Check learning rate, loss scaling, and data pipeline."
                )

        # Gradient clipping
        clip = self.config.gradient_clip_norm
        if clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                self._combined.parameters(), max_norm=clip
            ).item()
        else:
            grad_norm = sum(
                p.grad.norm().item() ** 2 for p in params_with_grad
            ) ** 0.5

        return grad_norm

    def _is_better(self, current: float, best: float) -> bool:
        if math.isnan(current):
            return False
        if self.config.early_stopping_mode == "min":
            return current < best
        return current > best

    def _save(self, epoch: int, suffix: str = "latest") -> None:
        path = Path(self.config.checkpoint_dir) / f"{self.experiment_name}_{suffix}.pt"
        self.save_checkpoint(str(path), epoch)

    def _resume(self, path: str) -> None:
        print(f"[Trainer] Resuming from: {path}")
        ckpt = load_checkpoint(path, self._combined, self.optimizer, device=self.device)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_val_metric = ckpt.get("best_val_metric", self.best_val_metric)
        self.history = ckpt.get("history", [])


# ──────────────────────────────────────────────────────────────────────────────
# Device helpers
# ──────────────────────────────────────────────────────────────────────────────

def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=torch.float32 if obj.is_floating_point() else obj.dtype)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _move_dict_to_device(d: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
