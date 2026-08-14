"""Trainer Engine for Latent Diffusion Model (PhotonShield AI V1.0).

Trains the lightweight temporal denoiser while keeping the PhotonV0 encoder frozen.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from module_05_latent_diffusion.latent_diffusion import LatentDiffusionModel


def plot_diffusion_curves(history: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot diffusion training and reconstruction loss curves."""
    epochs = [h["epoch"] for h in history]
    train_diff_loss = [h["train_diffusion_loss"] for h in history]
    val_diff_loss = [h["val_diffusion_loss"] for h in history]
    val_rec_mse = [h["val_reconstruction_mse"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#ffffff")

    # 1. Noise Prediction Loss
    axes[0].plot(epochs, train_diff_loss, label="Train Noise MSE", color="#1f77b4", lw=2)
    axes[0].plot(epochs, val_diff_loss, label="Val Noise MSE", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_title("Diffusion Noise Prediction Loss (MSE)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Latent Reconstruction Error
    axes[1].plot(epochs, val_rec_mse, label="Val Reconstruction MSE", color="#2ca02c", lw=2)
    axes[1].set_title("Latent Reconstruction MSE: MSE(Z_hat, Z_0)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


class DiffusionTrainer:
    """Trains and validates the conditional latent diffusion model."""

    def __init__(
        self,
        model: LatentDiffusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        cfg_train = config.get("training", {})
        self.epochs = int(cfg_train.get("epochs", 50))
        self.lr = float(cfg_train.get("learning_rate", 5e-4))
        self.weight_decay = float(cfg_train.get("weight_decay", 1e-4))
        self.patience = int(cfg_train.get("early_stopping_patience", 10))
        self.use_amp = bool(cfg_train.get("amp", True)) and (device.type == "cuda")

        # Train only the denoiser parameters
        self.optimizer = AdamW(self.model.denoiser.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-5)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.checkpoint_dir = Path("checkpoints/v1_diffusion")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path("results/photon_v1")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_epoch(self, dataloader: DataLoader, compute_full_reconstruction: bool = True) -> Tuple[float, float, float]:
        """Evaluate diffusion noise loss and latent reconstruction MSE."""
        self.model.denoiser.eval()
        total_diff_loss = 0.0
        rec_mse_sum = 0.0
        corrupted_mse_sum = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch["features"].to(self.device)
                B = x.shape[0]

                # 1. Diffusion noise prediction loss
                diff_loss, z_0, z_c, _ = self.model(x)
                total_diff_loss += diff_loss.item() * B

                # 2. Corrupted baseline MSE: MSE(Z_c, Z_0)
                corrupted_mse = nn.functional.mse_loss(z_c, z_0).item()
                corrupted_mse_sum += corrupted_mse * B

                # 3. Full reverse diffusion reconstruction MSE: MSE(Z_hat, Z_0)
                if compute_full_reconstruction:
                    z_hat = self.model.scheduler.reconstruct(
                        denoiser=self.model.denoiser,
                        condition=z_c,
                        num_inference_steps=min(20, self.model.timesteps),
                    )
                    rec_mse = nn.functional.mse_loss(z_hat, z_0).item()
                    rec_mse_sum += rec_mse * B

                total_samples += B

        mean_diff_loss = total_diff_loss / max(total_samples, 1)
        mean_corrupted_mse = corrupted_mse_sum / max(total_samples, 1)
        mean_rec_mse = (rec_mse_sum / max(total_samples, 1)) if compute_full_reconstruction else mean_diff_loss

        return mean_diff_loss, mean_corrupted_mse, mean_rec_mse

    def train(self) -> Dict[str, Any]:
        """Run full training loop with early stopping."""
        print(f"[DiffusionTrainer] Training lightweight denoiser on device: {self.device}")
        print(f"[DiffusionTrainer] Trainable Denoiser Parameters: {self.model.denoiser.count_parameters():,}")
        print(f"[DiffusionTrainer] Frozen Encoder Parameters: {sum(p.numel() for p in self.model.encoder.parameters()):,}")

        # Reset peak VRAM tracker
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        history = []
        best_val_rec_mse = float("inf")
        best_epoch = 0
        patience_counter = 0

        # CSV Logging
        csv_path = self.results_dir / "metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_diffusion_loss", "val_diffusion_loss", "val_corrupted_mse", "val_reconstruction_mse", "lr"])

        t_start = time.perf_counter()
        batch_latencies = []

        for epoch in range(1, self.epochs + 1):
            self.model.denoiser.train()
            train_diff_loss = 0.0
            total_train_samples = 0

            for batch in self.train_loader:
                t0 = time.perf_counter()
                x = batch["features"].to(self.device)
                B = x.shape[0]

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        loss, _, _, _ = self.model(x)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.denoiser.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, _, _, _ = self.model(x)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.denoiser.parameters(), max_norm=1.0)
                    self.optimizer.step()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                batch_latencies.append((time.perf_counter() - t0) * 1000.0)

                train_diff_loss += loss.item() * B
                total_train_samples += B

            curr_lr = float(self.scheduler.get_last_lr()[0])
            self.scheduler.step()
            train_diff_loss /= max(total_train_samples, 1)

            # Validation
            val_diff_loss, val_corr_mse, val_rec_mse = self.evaluate_epoch(self.val_loader, compute_full_reconstruction=True)

            row = {
                "epoch": epoch,
                "train_diffusion_loss": round(train_diff_loss, 5),
                "val_diffusion_loss": round(val_diff_loss, 5),
                "val_corrupted_mse": round(val_corr_mse, 5),
                "val_reconstruction_mse": round(val_rec_mse, 5),
                "lr": curr_lr,
            }
            history.append(row)

            # Append to CSV
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, row["train_diffusion_loss"], row["val_diffusion_loss"], row["val_corrupted_mse"], row["val_reconstruction_mse"], row["lr"]])

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_diff_loss:.5f} | Val Loss: {val_diff_loss:.5f} | "
                f"Val Base MSE: {val_corr_mse:.5f} -> Reconstructed MSE: {val_rec_mse:.5f}"
            )

            # Save Best Model Checkpoint
            if val_rec_mse < best_val_rec_mse:
                best_val_rec_mse = val_rec_mse
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.denoiser.state_dict(), self.checkpoint_dir / "best_diffusion.pt")
                print(f"  --> Saved new best diffusion checkpoint at epoch {epoch} (Val Rec MSE: {val_rec_mse:.5f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without reconstruction improvement.")
                    break

        total_time = time.perf_counter() - t_start
        torch.save(self.model.denoiser.state_dict(), self.checkpoint_dir / "last_diffusion.pt")

        # Plot curves
        plot_diffusion_curves(history, self.results_dir / "diffusion_training_curve.png")

        # Peak VRAM & batch latency
        peak_vram_gb = round(torch.cuda.max_memory_allocated(self.device) / (1024**3), 4) if self.device.type == "cuda" else 0.0
        avg_batch_lat = float(np.mean(batch_latencies)) if batch_latencies else 0.0

        return {
            "best_epoch": best_epoch,
            "best_val_rec_mse": best_val_rec_mse,
            "total_time_sec": round(total_time, 2),
            "peak_vram_gb": peak_vram_gb,
            "avg_batch_latency_ms": round(avg_batch_lat, 2),
            "history": history,
        }
