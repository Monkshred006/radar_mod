"""Trainer Engine for Latent Diffusion Model (PhotonShield AI V1).

Logs component losses (diffusion, x0 reconstruction, missing-frame) and validation metrics.
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
from module_05_latent_diffusion.losses import DiffusionLoss


def plot_diffusion_curves(history: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot diffusion training and reconstruction loss curves."""
    epochs = [h["epoch"] for h in history]
    train_total_loss = [h["train_total_loss"] for h in history]
    train_diff_loss = [h["train_diff_loss"] for h in history]
    train_missing_loss = [h["train_missing_loss"] for h in history]
    val_rec_mse = [h["val_reconstructed_mse"] for h in history]
    val_missing_mse = [h["val_missing_mse"] for h in history]
    val_corr_mse = [h["val_corrupted_mse"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#ffffff")

    # 1. Training Component Losses
    axes[0].plot(epochs, train_total_loss, label="Total Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, train_diff_loss, label="Noise Loss (L_diff)", color="#ff7f0e", lw=1.5, linestyle="--")
    axes[0].plot(epochs, train_missing_loss, label="Missing Loss (L_miss)", color="#d62728", lw=1.5, linestyle=":")
    axes[0].set_title("Training Component Losses", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Validation Imputation MSE
    axes[1].plot(epochs, val_corr_mse, label="Corrupted Baseline MSE", color="#7f7f7f", lw=1.5, linestyle="--")
    axes[1].plot(epochs, val_rec_mse, label="Full Reconstructed MSE", color="#2ca02c", lw=2)
    axes[1].plot(epochs, val_missing_mse, label="Missing-Frame MSE", color="#9467bd", lw=2)
    axes[1].set_title("Validation Latent Imputation MSE", fontsize=11, fontweight="bold")
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

    def evaluate_epoch(self, dataloader: DataLoader, compute_full_reconstruction: bool = True) -> Dict[str, float]:
        """Evaluate validation metrics across dataloader."""
        self.model.denoiser.eval()

        total_samples = 0
        total_corr_mse = 0.0
        total_rec_mse = 0.0
        total_missing_mse = 0.0
        total_obs_mse = 0.0

        with torch.no_grad():
            for batch in dataloader:
                x = batch["features"].to(self.device)
                B = x.shape[0]

                # Full conditional reconstruction
                z_hat, z_0, z_c, mask = self.model.reconstruct(
                    x=x,
                    num_steps=min(20, self.model.timesteps),
                )

                corr_metrics = DiffusionLoss.reconstruction_metrics(z_c, z_0, mask)
                rec_metrics = DiffusionLoss.reconstruction_metrics(z_hat, z_0, mask)

                total_corr_mse += corr_metrics["full_mse"] * B
                total_rec_mse += rec_metrics["full_mse"] * B
                total_missing_mse += rec_metrics["missing_mse"] * B
                total_obs_mse += rec_metrics["observed_mse"] * B
                total_samples += B

        mean_corr = total_corr_mse / max(total_samples, 1)
        mean_rec = total_rec_mse / max(total_samples, 1)
        mean_miss = total_missing_mse / max(total_samples, 1)
        mean_obs = total_obs_mse / max(total_samples, 1)
        improvement = 100.0 * (mean_corr - mean_rec) / max(mean_corr, 1e-8)

        return {
            "corrupted_mse": mean_corr,
            "reconstructed_mse": mean_rec,
            "missing_mse": mean_miss,
            "observed_mse": mean_obs,
            "improvement": improvement,
        }

    def train(self) -> Dict[str, Any]:
        """Run full training loop with early stopping."""
        print(f"[DiffusionTrainer] Training on device: {self.device}")
        print(f"[DiffusionTrainer] Trainable Denoiser Parameters: {self.model.denoiser.count_parameters():,}")
        print(f"[DiffusionTrainer] Frozen Encoder Parameters: {sum(p.numel() for p in self.model.encoder.parameters()):,}")

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
            writer.writerow([
                "epoch", "train_total_loss", "train_diff_loss", "train_recon_loss", "train_missing_loss",
                "val_corrupted_mse", "val_reconstructed_mse", "val_missing_mse", "val_observed_mse", "val_improvement", "lr"
            ])

        t_start = time.perf_counter()
        batch_latencies = []

        for epoch in range(1, self.epochs + 1):
            self.model.denoiser.train()
            sum_total_loss = 0.0
            sum_diff_loss = 0.0
            sum_recon_loss = 0.0
            sum_missing_loss = 0.0
            total_train_samples = 0

            for batch in self.train_loader:
                t0 = time.perf_counter()
                x = batch["features"].to(self.device)
                B = x.shape[0]

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        loss, loss_dict, _, _, _, _ = self.model(x)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.denoiser.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, loss_dict, _, _, _, _ = self.model(x)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.denoiser.parameters(), max_norm=1.0)
                    self.optimizer.step()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                batch_latencies.append((time.perf_counter() - t0) * 1000.0)

                sum_total_loss += loss.item() * B
                sum_diff_loss += loss_dict["diff_loss"] * B
                sum_recon_loss += loss_dict["recon_loss"] * B
                sum_missing_loss += loss_dict["missing_loss"] * B
                total_train_samples += B

            curr_lr = float(self.scheduler.get_last_lr()[0])
            self.scheduler.step()

            mean_train_total = sum_total_loss / max(total_train_samples, 1)
            mean_train_diff = sum_diff_loss / max(total_train_samples, 1)
            mean_train_recon = sum_recon_loss / max(total_train_samples, 1)
            mean_train_missing = sum_missing_loss / max(total_train_samples, 1)

            # Validation
            val_metrics = self.evaluate_epoch(self.val_loader, compute_full_reconstruction=True)

            row = {
                "epoch": epoch,
                "train_total_loss": round(mean_train_total, 5),
                "train_diff_loss": round(mean_train_diff, 5),
                "train_recon_loss": round(mean_train_recon, 5),
                "train_missing_loss": round(mean_train_missing, 5),
                "val_corrupted_mse": round(val_metrics["corrupted_mse"], 5),
                "val_reconstructed_mse": round(val_metrics["reconstructed_mse"], 5),
                "val_missing_mse": round(val_metrics["missing_mse"], 5),
                "val_observed_mse": round(val_metrics["observed_mse"], 5),
                "val_improvement": round(val_metrics["improvement"], 2),
                "lr": curr_lr,
            }
            history.append(row)

            # Append to CSV
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, row["train_total_loss"], row["train_diff_loss"], row["train_recon_loss"], row["train_missing_loss"],
                    row["val_corrupted_mse"], row["val_reconstructed_mse"], row["val_missing_mse"], row["val_observed_mse"], row["val_improvement"], row["lr"]
                ])

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Total: {mean_train_total:.4f} (Diff: {mean_train_diff:.4f}, Rec: {mean_train_recon:.4f}, Miss: {mean_train_missing:.4f}) | "
                f"Val Base MSE: {val_metrics['corrupted_mse']:.4f} -> Rec MSE: {val_metrics['reconstructed_mse']:.4f} (Miss: {val_metrics['missing_mse']:.4f}, Imprv: {val_metrics['improvement']:.1f}%)"
            )

            # Save Best Checkpoint based on Validation Reconstruction MSE
            if val_metrics["reconstructed_mse"] < best_val_rec_mse:
                best_val_rec_mse = val_metrics["reconstructed_mse"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.denoiser.state_dict(), self.checkpoint_dir / "best_diffusion.pt")
                print(f"  --> Saved new best diffusion checkpoint at epoch {epoch} (Val Rec MSE: {best_val_rec_mse:.5f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without reconstruction improvement.")
                    break

        total_time = time.perf_counter() - t_start
        torch.save(self.model.denoiser.state_dict(), self.checkpoint_dir / "last_diffusion.pt")

        plot_diffusion_curves(history, self.results_dir / "diffusion_training_curve.png")

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
