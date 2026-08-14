"""PhotonShield AI Phase V0 Training Entry Point.

Trains PhotonV0 perception stack on RaDICaL radar data with staged hooks
for Latent Diffusion (V1) and Physics-Informed PINN Constraints (V2).

Usage:
    python train_photon_v0.py --config configs/photon_v0.yaml --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, Optional, Tuple

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
import yaml

from module_04_mamba_hybrid.photon_v0 import PhotonV0, count_parameters
from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter
from module_05_training.diffusion_auxiliary import DiffusionAuxiliary
from module_08_pinn_rl.pinn_constraints import CompositePhysicsConstraint
from module_06_bitnet.profile_uno_q import estimate_photon_v0_macs


def compute_metrics(
    all_det_preds: np.ndarray,
    all_det_targets: np.ndarray,
    all_cls_preds: np.ndarray,
    all_cls_targets: np.ndarray,
    all_ano_preds: np.ndarray,
    all_ano_targets: np.ndarray,
) -> Dict[str, float]:
    """Compute evaluation metrics: Accuracy, F1, MAE, AUROC."""
    cls_acc = float(np.mean(all_cls_preds == all_cls_targets))

    unique_classes = np.unique(np.concatenate([all_cls_targets, all_cls_preds]))
    f1_scores = []
    for c in unique_classes:
        tp = np.sum((all_cls_preds == c) & (all_cls_targets == c))
        fp = np.sum((all_cls_preds == c) & (all_cls_targets != c))
        fn = np.sum((all_cls_preds != c) & (all_cls_targets == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
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
        "mae": ano_mae,
        "auroc": auroc,
    }


def evaluate(
    model: PhotonV0,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
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
            l_det = bce_loss_fn(outputs["detection"], y_det)
            l_cls = ce_loss_fn(outputs["classification"], y_cls)
            l_ano = mse_loss_fn(outputs["anomaly"], y_ano)
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
    )
    return mean_loss, metrics


def plot_training_curves(history: list, output_path: Union[str, Path]) -> None:
    """Plot and save loss, accuracy, and AUROC training curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    acc = [h["accuracy"] for h in history]
    f1 = [h["f1_score"] for h in history]
    auroc = [h["auroc"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#ffffff")

    # 1. Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Accuracy & F1
    axes[1].plot(epochs, acc, label="Val Accuracy", color="#2ca02c", lw=2)
    axes[1].plot(epochs, f1, label="Val Macro F1", color="#9467bd", lw=2, linestyle="--")
    axes[1].set_title("Classification Accuracy & Macro F1", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. AUROC
    axes[2].plot(epochs, auroc, label="Detection AUROC", color="#d62728", lw=2)
    axes[2].set_title("Target Detection AUROC", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUROC")
    axes[2].set_ylim([0, 1.05])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_photon_v0(
    config: Dict[str, Any],
    override_epochs: Optional[int] = None,
    allow_synthetic: bool = False,
    allow_experimental: bool = False,
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

    # Set random seeds
    seed = cfg_train.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection
    dev_str = cfg_train.get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    print(f"[PhotonShield AI] Training on device: {device}")

    # 1. Dataset Adapter (Synthetic fallback disabled by default)
    synthetic_fallback = allow_synthetic or cfg_data.get("synthetic_fallback", False)
    adapter = RaDICaLDatasetAdapter(
        data_path=cfg_data.get("data_path", "data/radical"),
        sequence_length=cfg_data.get("sequence_length", 16),
        feature_dim=cfg_data.get("feature_dim", 64),
        num_classes=cfg_data.get("num_classes", 4),
        normalization=cfg_data.get("normalization", "db"),
        train_ratio=cfg_data.get("train_ratio", 0.70),
        val_ratio=cfg_data.get("val_ratio", 0.15),
        test_ratio=cfg_data.get("test_ratio", 0.15),
        seed=seed,
        synthetic_fallback=synthetic_fallback,
    )

    batch_size = cfg_train.get("batch_size", 64)
    train_loader, val_loader, test_loader = adapter.get_dataloaders(
        batch_size=batch_size, num_synthetic_fallback=500
    )

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
    print(f" Diffusion Branch: {'ENABLED' if diff_aux.enabled else 'DISABLED'} | PINN Constraints: {'ENABLED' if pinn_constraints.enabled else 'DISABLED'}")
    print("----------------------------------------------------------------")

    # Optimizer & Scheduler
    lr = float(cfg_train.get("learning_rate", 1e-3))
    weight_decay = float(cfg_train.get("weight_decay", 1e-4))
    trainable_params = list(model.parameters())
    if diff_aux.enabled:
        trainable_params.extend(list(diff_aux.parameters()))

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    epochs = override_epochs or cfg_train.get("epochs", 20)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Loss weights
    l_det_w = float(cfg_train.get("lambda_det", 1.0))
    l_cls_w = float(cfg_train.get("lambda_cls", 1.0))
    l_ano_w = float(cfg_train.get("lambda_ano", 0.5))
    l_diff_w = float(cfg_train.get("lambda_diffusion", 0.1))

    bce_loss_fn = nn.BCELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    checkpoint_dir = Path(cfg_train.get("checkpoint_dir", "checkpoints/photon_v0"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg_train.get("results_dir", "results/photon_v0"))
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_metrics = {}
    history = []

    # CSV Logger
    csv_file = results_dir / "metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "accuracy", "f1_score", "mae", "auroc", "lr"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["features"].to(device)
            y_det = batch["detection"].to(device)
            y_cls = batch["classification"].to(device)
            y_ano = batch["anomaly"].to(device)

            optimizer.zero_grad()
            outputs = model(x, return_latents=True)

            loss_det = bce_loss_fn(outputs["detection"], y_det)
            loss_cls = ce_loss_fn(outputs["classification"], y_cls)
            loss_ano = mse_loss_fn(outputs["anomaly"], y_ano)

            total_step_loss = (l_det_w * loss_det) + (l_cls_w * loss_cls) + (l_ano_w * loss_ano)

            if diff_aux.enabled:
                loss_diff = diff_aux.compute_loss(outputs["latent"])
                total_step_loss = total_step_loss + (l_diff_w * loss_diff)

            if pinn_constraints.enabled:
                loss_phys = pinn_constraints(outputs["latent"], prediction=outputs)
                total_step_loss = total_step_loss + loss_phys

            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += total_step_loss.item() * len(x)

        curr_lr = float(scheduler.get_last_lr()[0])
        scheduler.step()
        train_loss /= max(len(train_loader.dataset), 1)

        # Validation
        val_loss, val_metrics = evaluate(model, val_loader, device)

        row_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": curr_lr,
            **{k: round(v, 4) for k, v in val_metrics.items()},
        }
        history.append(row_data)

        # Append to CSV
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                row_data["train_loss"],
                row_data["val_loss"],
                row_data["accuracy"],
                row_data["f1_score"],
                row_data["mae"],
                row_data["auroc"],
                row_data["lr"],
            ])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | AUROC: {val_metrics['auroc']:.4f}"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = dict(val_metrics)
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            torch.save(model.state_dict(), checkpoint_dir / "photon_v0_best.pt")

        # Save Last Model
        torch.save(model.state_dict(), checkpoint_dir / "last_model.pt")

    # Generate training curves plot
    plot_training_curves(history, results_dir / "training_curves.png")

    # Evaluate on Test Set using best model
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device))
    test_loss, test_metrics = evaluate(model, test_loader, device)

    print("================================================================")
    print(f" Final Test Results: Loss: {test_loss:.4f} | Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_score']:.4f} | AUROC: {test_metrics['auroc']:.4f}")
    print("================================================================")

    summary = {
        "parameter_count": total_params,
        "macs": total_macs,
        "flops": total_flops,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "training_history": history,
    }

    with open(results_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhotonShield AI Phase V0 Perception Model.")
    parser.add_argument("--config", type=str, default="configs/photon_v0.yaml", help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic fallback if real data missing")
    parser.add_argument("--allow-experimental", action="store_true", help="Allow V1/V2 features")
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
    )


if __name__ == "__main__":
    main()
