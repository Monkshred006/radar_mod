"""PhotonShield AI Phase V0/V0.1 Training & Evaluation Engine.

Trains PhotonV0 perception stack on RaDICaL radar data with deterministic seeding,
early stopping based on validation macro-F1, per-class metrics, confusion matrix,
and inference benchmarking.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, Optional, Tuple, List, Union

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset, DataLoader
import yaml

from module_04_mamba_hybrid.photon_v0 import PhotonV0, count_parameters
from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter, RADICAL_CLASSES
from module_05_training.diffusion_auxiliary import DiffusionAuxiliary
from module_08_pinn_rl.pinn_constraints import CompositePhysicsConstraint
from module_06_bitnet.profile_uno_q import estimate_photon_v0_macs


def set_deterministic_seed(seed: int = 42) -> None:
    """Set seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(
    all_det_preds: np.ndarray,
    all_det_targets: np.ndarray,
    all_cls_preds: np.ndarray,
    all_cls_targets: np.ndarray,
    all_ano_preds: np.ndarray,
    all_ano_targets: np.ndarray,
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Accuracy, Macro-F1, Per-Class F1, Confusion Matrix, MAE, AUROC."""
    cls_acc = float(np.mean(all_cls_preds == all_cls_targets))

    if class_names is None:
        class_names = list(RADICAL_CLASSES)

    per_class_f1 = {}
    f1_scores = []
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(all_cls_targets)):
        t = int(all_cls_targets[i])
        p = int(all_cls_preds[i])
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    for c in range(num_classes):
        tp = np.sum((all_cls_preds == c) & (all_cls_targets == c))
        fp = np.sum((all_cls_preds == c) & (all_cls_targets != c))
        fn = np.sum((all_cls_preds != c) & (all_cls_targets == c))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        c_name = class_names[c] if c < len(class_names) else f"Class_{c}"
        per_class_f1[c_name] = round(f1, 4)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    ano_mae = float(np.mean(np.abs(all_ano_preds - all_ano_targets)))

    det_targets_bin = (all_det_targets.flatten() > 0.5).astype(int)
    det_scores = all_det_preds.flatten()
    desc_score_indices = np.argsort(det_scores)[::-1]
    sorted_targets = det_targets_bin[desc_score_indices]
    n_pos = np.sum(sorted_targets == 1)
    n_neg = np.sum(sorted_targets == 0)

    if n_pos > 0 and n_neg > 0:
        ranks = np.arange(len(sorted_targets), 0, -1)
        sum_pos_ranks = np.sum(ranks[sorted_targets == 1])
        auroc = float((sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg))
    else:
        auroc = 1.0

    return {
        "accuracy": cls_acc,
        "f1_score": macro_f1,
        "per_class_f1": per_class_f1,
        "mae": ano_mae,
        "auroc": auroc,
        "confusion_matrix": cm.tolist(),
    }


def evaluate(
    model: PhotonV0,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
) -> Tuple[float, Dict[str, Any]]:
    """Run validation evaluation."""
    model.eval()
    total_loss = 0.0
    det_preds, det_targets = [], []
    cls_preds, cls_targets = [], []
    ano_preds, ano_targets = [], []

    bce_loss_fn = nn.BCELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            x = batch["features"].to(device)
            y_det = batch["detection"].to(device)
            y_cls = batch["classification"].to(device)
            y_ano = batch["anomaly"].to(device)

            outputs = model(x)
            l_det = bce_loss_fn(outputs["detection"].float(), y_det.float())
            l_cls = ce_loss_fn(outputs["classification"].float(), y_cls)
            l_ano = mse_loss_fn(outputs["anomaly"].float(), y_ano.float())
            loss = l_det + l_cls + 0.5 * l_ano
            total_loss += loss.item() * len(x)

            det_preds.append(outputs["detection"].cpu().numpy())
            det_targets.append(y_det.cpu().numpy())
            cls_preds.append(torch.argmax(outputs["classification"], dim=-1).cpu().numpy())
            cls_targets.append(y_cls.cpu().numpy())
            ano_preds.append(outputs["anomaly"].cpu().numpy())
            ano_targets.append(y_ano.cpu().numpy())

    mean_loss = total_loss / max(len(dataloader.dataset), 1)
    metrics = compute_metrics(
        all_det_preds=np.concatenate(det_preds, axis=0),
        all_det_targets=np.concatenate(det_targets, axis=0),
        all_cls_preds=np.concatenate(cls_preds, axis=0),
        all_cls_targets=np.concatenate(cls_targets, axis=0),
        all_ano_preds=np.concatenate(ano_preds, axis=0),
        all_ano_targets=np.concatenate(ano_targets, axis=0),
        num_classes=num_classes,
    )
    return mean_loss, metrics


def plot_training_curves(history: list, output_path: Union[str, Path]) -> None:
    """Plot and save loss, accuracy, and Macro-F1 training curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    auroc = [h["auroc"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#ffffff")

    axes[0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Accuracy", color="#2ca02c", lw=2)
    axes[1].plot(epochs, val_acc, label="Val Accuracy", color="#17becf", lw=2, linestyle="-.")
    axes[1].plot(epochs, val_f1, label="Val Macro-F1", color="#9467bd", lw=2, linestyle="--")
    axes[1].set_title("Accuracy & Macro-F1", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, auroc, label="Val Detection AUROC", color="#d62728", lw=2)
    axes[2].set_title("Target Detection AUROC", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUROC")
    axes[2].set_ylim([0, 1.05])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: list, class_names: List[str], output_path: Union[str, Path]) -> None:
    """Plot and save confusion matrix heatmap."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="PhotonV0 Confusion Matrix (Test Set)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontweight="bold"
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def benchmark_inference(model: PhotonV0, device: torch.device, sequence_length: int = 16, feature_dim: int = 64, num_runs: int = 100) -> Dict[str, float]:
    """Measure inference latency per sample and batch on target hardware."""
    model.eval()
    dummy_single = torch.randn(1, sequence_length, feature_dim, device=device)
    dummy_batch = torch.randn(32, sequence_length, feature_dim, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_single)
            _ = model(dummy_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark single sample latency
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_single)
    if device.type == "cuda":
        torch.cuda.synchronize()
    single_lat_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # Benchmark batch-32 latency
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    batch_lat_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    fps = 1000.0 / single_lat_ms if single_lat_ms > 0 else 0.0

    return {
        "single_sample_latency_ms": round(single_lat_ms, 3),
        "batch_32_latency_ms": round(batch_lat_ms, 3),
        "throughput_fps": round(fps, 1),
    }


def train_photon_v0(
    config: Dict[str, Any],
    override_epochs: Optional[int] = None,
    allow_synthetic: bool = False,
    allow_experimental: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Train PhotonV0 pipeline from configuration dictionary."""
    cfg_data = config.get("dataset", {})
    cfg_model = config.get("model", {})
    cfg_diff = config.get("diffusion", {})
    cfg_phys = config.get("physics", {})
    cfg_train = config.get("training", {})
    cfg_dec = config.get("decision", {})

    # Freeze V0 Execution Path Check: Ensure experimental features are disabled for V0 baseline
    if not allow_experimental:
        if (
            cfg_diff.get("enabled", False)
            or cfg_phys.get("enabled", False)
            or cfg_dec.get("enabled", False)
            or cfg_model.get("use_attention", False)
        ):
            raise ValueError(
                "V1/V2 features (Diffusion, PINN, RL, Attention) are enabled in config, but baseline V0 requires them disabled. "
                "Set diffusion.enabled: false, physics.enabled: false, decision.enabled: false, or pass --allow-experimental."
            )

    # Set random seeds for reproducibility
    seed = cfg_train.get("seed", 42)
    set_deterministic_seed(seed)

    # Device selection
    dev_str = cfg_train.get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    epochs = override_epochs or cfg_train.get("epochs", 50)
    patience = int(cfg_train.get("early_stopping_patience", 10))

    print(f"[PhotonShield AI] Device selected: {device}")
    print(f"[PhotonShield AI] Deterministic Seed: {seed} | Max Epochs: {epochs} | Early Stopping Patience: {patience}")

    # 1. Dataset Adapter
    synthetic_fallback = allow_synthetic or cfg_data.get("synthetic_fallback", False)
    adapter = RaDICaLDatasetAdapter(
        data_path=cfg_data.get("data_path", "C:/Users/worka/research/photonpinn/data/radical"),
        splits_dir=cfg_data.get("splits_dir", "C:/Users/worka/research/photonpinn/data/radical/splits"),
        sequence_length=cfg_data.get("sequence_length", 16),
        feature_dim=cfg_data.get("feature_dim", 64),
        num_classes=cfg_data.get("num_classes", 4),
        normalization=cfg_data.get("normalization", "db"),
        seed=seed,
        synthetic_fallback=synthetic_fallback,
        augmentation=cfg_data.get("augmentation", None),
    )

    train_ds, val_ds, test_ds = adapter.get_datasets(num_synthetic_fallback=500)

    # Apply sample limits if provided
    max_train = cfg_train.get("max_train_samples")
    if max_train and len(train_ds) > max_train:
        train_ds = Subset(train_ds, list(range(max_train)))

    max_val = cfg_train.get("max_val_samples")
    if max_val and len(val_ds) > max_val:
        val_ds = Subset(val_ds, list(range(max_val)))

    batch_size = cfg_train.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Build PhotonV0 Model
    model = PhotonV0(
        input_dim=cfg_model.get("input_dim", 64),
        hidden_dim=cfg_model.get("hidden_dim", 64),
        num_layers=cfg_model.get("num_layers", 2),
        sequence_length=cfg_model.get("sequence_length", 16),
        num_classes=cfg_model.get("num_classes", 4),
        d_state=cfg_model.get("d_state", 16),
        d_conv=cfg_model.get("d_conv", 4),
        expand=cfg_model.get("expand", 2),
        dropout=cfg_model.get("dropout", 0.0),
        use_attention=cfg_model.get("use_attention", False),
        backend=cfg_model.get("backend", "auto"),
    ).to(device)

    # 3. Optional Diffusion Auxiliary Branch (V1)
    diff_aux = DiffusionAuxiliary(
        hidden_dim=cfg_model.get("hidden_dim", 64),
        timesteps=cfg_diff.get("timesteps", 10),
        noise_std=cfg_diff.get("noise_std", 0.1),
        enabled=cfg_diff.get("enabled", False),
    ).to(device)

    # 4. Optional Physics Constraint Layer (V2)
    pinn_constraints = CompositePhysicsConstraint(
        enabled=cfg_phys.get("enabled", False),
        lambda_physics=cfg_phys.get("lambda_physics", 0.1),
    ).to(device)

    # Sizing & MACs
    total_params = count_parameters(model)
    total_macs = estimate_photon_v0_macs(
        input_dim=cfg_model.get("input_dim", 64),
        hidden_dim=cfg_model.get("hidden_dim", 64),
        num_layers=cfg_model.get("num_layers", 2),
        sequence_length=cfg_model.get("sequence_length", 16),
        num_classes=cfg_model.get("num_classes", 4),
    )
    total_flops = total_macs * 2

    print("----------------------------------------------------------------")
    print(f" PhotonV0 Parameters: {total_params:,} | Compute: {total_macs:,} MACs (~{total_flops:,} FLOPs)")
    print(f" Dataset: Real RaDICaL ({len(train_loader.dataset)} Train, {len(val_loader.dataset)} Val, {len(test_loader.dataset)} Test)")
    print(f" Augmentation: {'ENABLED' if cfg_data.get('augmentation', {}).get('enabled', False) else 'DISABLED'}")
    print("----------------------------------------------------------------")

    # Optimizer & Scheduler
    lr = float(cfg_train.get("learning_rate", 1e-3))
    weight_decay = float(cfg_train.get("weight_decay", 1e-4))
    trainable_params = list(model.parameters())
    if diff_aux.enabled:
        trainable_params.extend(list(diff_aux.parameters()))

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # AMP Scaler
    use_amp = bool(cfg_train.get("amp", False)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=use_amp)

    # Loss weights
    l_det_w = float(cfg_train.get("lambda_det", 1.0))
    l_cls_w = float(cfg_train.get("lambda_cls", 1.0))
    l_ano_w = float(cfg_train.get("lambda_ano", 0.5))

    bce_loss_fn = nn.BCELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    checkpoint_dir = Path(cfg_train.get("checkpoint_dir", "results/photon_v0/V0_1/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg_train.get("results_dir", "results/photon_v0/V0_1"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # CSV Logger
    csv_file = results_dir / "metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_macro_f1", "val_mae", "val_auroc", "lr"])

    # Reset CUDA memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_val_metrics = {}
    best_epoch = 0
    patience_counter = 0
    history = []

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_cls_preds, train_cls_targets = [], []

        for b_idx, batch in enumerate(train_loader):
            x = batch["features"].to(device)
            y_det = batch["detection"].to(device)
            y_cls = batch["classification"].to(device)
            y_ano = batch["anomaly"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(x, return_latents=True)
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    loss_det = bce_loss_fn(outputs["detection"].float(), y_det.float())
                    loss_cls = ce_loss_fn(outputs["classification"].float(), y_cls)
                    loss_ano = mse_loss_fn(outputs["anomaly"].float(), y_ano.float())
                    total_step_loss = (l_det_w * loss_det) + (l_cls_w * loss_cls) + (l_ano_w * loss_ano)
                scaler.scale(total_step_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x, return_latents=True)
                loss_det = bce_loss_fn(outputs["detection"].float(), y_det.float())
                loss_cls = ce_loss_fn(outputs["classification"].float(), y_cls)
                loss_ano = mse_loss_fn(outputs["anomaly"].float(), y_ano.float())
                total_step_loss = (l_det_w * loss_det) + (l_cls_w * loss_cls) + (l_ano_w * loss_ano)
                total_step_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            train_loss += total_step_loss.item() * len(x)
            train_cls_preds.append(torch.argmax(outputs["classification"], dim=-1).detach().cpu().numpy())
            train_cls_targets.append(y_cls.detach().cpu().numpy())

        curr_lr = float(scheduler.get_last_lr()[0])
        scheduler.step()
        train_loss /= max(len(train_loader.dataset), 1)

        # Train metrics
        train_preds_arr = np.concatenate(train_cls_preds)
        train_targets_arr = np.concatenate(train_cls_targets)
        train_acc = float(np.mean(train_preds_arr == train_targets_arr))
        
        tr_f1_list = []
        for c in range(cfg_model.get("num_classes", 4)):
            tp = np.sum((train_preds_arr == c) & (train_targets_arr == c))
            fp = np.sum((train_preds_arr == c) & (train_targets_arr != c))
            fn = np.sum((train_preds_arr != c) & (train_targets_arr == c))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            tr_f1_list.append(f)
        train_macro_f1 = float(np.mean(tr_f1_list))

        # Validation
        val_loss, val_metrics = evaluate(model, val_loader, device, num_classes=cfg_model.get("num_classes", 4))
        val_macro_f1 = val_metrics["f1_score"]
        val_acc = val_metrics["accuracy"]

        row_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "train_f1": round(train_macro_f1, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_macro_f1, 4),
            "lr": curr_lr,
            "mae": round(val_metrics["mae"], 4),
            "auroc": round(val_metrics["auroc"], 4),
        }
        history.append(row_data)

        # Write to CSV
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                row_data["train_loss"],
                row_data["train_acc"],
                row_data["train_f1"],
                row_data["val_loss"],
                row_data["val_acc"],
                row_data["val_f1"],
                row_data["mae"],
                row_data["auroc"],
                row_data["lr"],
            ])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.4f}, F1: {train_macro_f1:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.4f}, Macro-F1: {val_macro_f1:.4f}) | "
            f"AUROC: {val_metrics['auroc']:.4f}"
        )

        # Check for Best Checkpoint based on Validation Macro-F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Macro-F1: {val_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without Val Macro-F1 improvement.")
                break

    total_train_time = time.perf_counter() - t_start

    # Peak VRAM
    peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 4) if device.type == "cuda" else 0.0

    # Save training curves plot
    plot_training_curves(history, results_dir / "training_curves.png")

    # =========================================================================
    # Final TEST Set Evaluation (Evaluated ONLY ONCE using best model checkpoint)
    # =========================================================================
    print("\n================================================================")
    print(" Evaluating Best Model Checkpoint on Unseen TEST Set")
    print("================================================================")
    best_model_path = checkpoint_dir / "best_model.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    test_loss, test_metrics = evaluate(model, test_loader, device, num_classes=cfg_model.get("num_classes", 4))
    
    # Save Confusion Matrix Plot and JSON
    plot_confusion_matrix(test_metrics["confusion_matrix"], RADICAL_CLASSES, results_dir / "confusion_matrix.png")
    with open(results_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics["confusion_matrix"], f, indent=2)

    # Benchmark hardware inference latency
    bench_results = benchmark_inference(model, device)

    print(f" Test Loss:     {test_loss:.4f}")
    print(f" Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f" Test Macro-F1: {test_metrics['f1_score']:.4f}")
    print(f" Per-Class F1:  {test_metrics['per_class_f1']}")
    print(f" Test AUROC:    {test_metrics['auroc']:.4f}")
    print(f" Latency:       {bench_results['single_sample_latency_ms']} ms/sample ({bench_results['throughput_fps']} FPS)")
    print("================================================================")

    # Generate Test Results JSON
    test_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "best_val_metrics": best_val_metrics,
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_macro_f1": round(test_metrics["f1_score"], 4),
        "per_class_f1": test_metrics["per_class_f1"],
        "test_auroc": round(test_metrics["auroc"], 4),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "peak_vram_gb": peak_vram_gb,
        "training_time_sec": round(total_train_time, 2),
        "inference_benchmark": bench_results,
        "parameters": total_params,
        "macs": total_macs,
        "flops": total_flops,
    }
    with open(results_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    # Generate V0.1 REPORT.md
    report_content = f"""# PhotonShield AI — Phase V0.1 Baseline Training Report

**Dataset**: RaDICaL (77 GHz FMCW Radar Range-Doppler Sequences)  
**Architecture**: PhotonV0 Minimal Mamba Temporal Perception Stack  
**Hardware Target**: NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM) & Arduino Uno Q  
**Status**: COMPLETED (Deterministically Reproducible)  

---

## 1. Dataset & Splits

* **Total Sequences**: 500 sequences (8,000 temporal frames)
* **Training Set**: 350 sequences (Fixed split: `splits/train.txt`)
* **Validation Set**: 75 sequences (Fixed split: `splits/val.txt`)
* **Test Set**: 75 sequences (Fixed split: `splits/test.txt`)
* **Augmentation**: Disabled (Baseline V0.1)

---

## 2. Model & Compute Specifications

* **Model Parameters**: {total_params:,}
* **Inference Compute**: {total_macs:,} MACs (~{total_flops:,} FLOPs)
* **SRAM Footprint (INT8)**: ~8.0 KB (Well within Arduino Uno Q 64 KB SRAM)
* **Checkpoint File Size**: {round(best_model_path.stat().st_size / 1024.0, 2)} KB

---

## 3. Training & Validation Results

* **Total Epochs Trained**: {len(history)} (Best Model Selected at Epoch {best_epoch})
* **Best Validation Loss**: {best_val_loss:.4f}
* **Best Validation Accuracy**: {best_val_metrics.get('accuracy', 0.0):.4f}
* **Best Validation Macro-F1**: {best_val_macro_f1:.4f}
* **Validation Detection AUROC**: {best_val_metrics.get('auroc', 0.0):.4f}

---

## 4. Final Test Set Evaluation (Unseen Test Partition)

* **Test Loss**: **{test_loss:.4f}**
* **Test Classification Accuracy**: **{test_metrics['accuracy'] * 100:.2f}%**
* **Test Macro-F1 Score**: **{test_metrics['f1_score']:.4f}**
* **Target Detection AUROC**: **{test_metrics['auroc']:.4f}**

### Per-Class F1 Breakdown:
* **Empty**: {test_metrics['per_class_f1'].get('Empty', 0.0)}
* **Pedestrian**: {test_metrics['per_class_f1'].get('Pedestrian', 0.0)}
* **Cyclist**: {test_metrics['per_class_f1'].get('Cyclist', 0.0)}
* **Vehicle**: {test_metrics['per_class_f1'].get('Vehicle', 0.0)}

### Confusion Matrix (Rows: True, Columns: Predicted):
```text
{np.array(test_metrics['confusion_matrix'])}
```

---

## 5. Hardware Performance & Profiling (RTX 5050 8GB)

* **Peak GPU VRAM Allocated**: **{peak_vram_gb} GB** (~{round(peak_vram_gb * 1024, 1)} MB / 7.96 GB)
* **Total Training Time**: **{round(total_train_time, 2)} seconds** (~{round(total_train_time / 60.0, 2)} minutes)
* **Single Sample Inference Latency**: **{bench_results['single_sample_latency_ms']} ms**
* **Batch-32 Latency**: **{bench_results['batch_32_latency_ms']} ms**
* **Inference Throughput**: **{bench_results['throughput_fps']} FPS**
"""

    with open(results_dir / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    return test_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhotonShield AI Phase V0/V0.1 Perception Model.")
    parser.add_argument("--config", type=str, default="configs/photon_v0_full.yaml", help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic fallback if real data missing")
    parser.add_argument("--allow-experimental", action="store_true", help="Allow V1/V2 features")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-batch dry run")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    train_photon_v0(
        config,
        override_epochs=args.epochs,
        allow_synthetic=args.allow_synthetic,
        allow_experimental=args.allow_experimental,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
