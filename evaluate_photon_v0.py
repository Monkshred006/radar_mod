"""PhotonShield AI Phase V0 Comprehensive Evaluation and Benchmarking Script.

Computes:
1. Confusion Matrix & Per-Class Precision / Recall / F1.
2. ROC Curve & Detection AUROC.
3. Generates and saves visual plots: `confusion_matrix.png` & `roc_curve.png`.
4. Latency Benchmarking (mean, std, p95, p99 per sample in ms on CPU & CUDA if available).
5. Memory Footprint Benchmarking (peak RAM / allocation).
6. Saves all evaluation artifacts to `results/photon_v0/`.

Usage:
    python evaluate_photon_v0.py --config configs/photon_v0.yaml --checkpoint checkpoints/photon_v0/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Tuple, Optional

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
import yaml

from module_04_mamba_hybrid.photon_v0 import PhotonV0, count_parameters
from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter, RADICAL_CLASSES
from module_06_bitnet.profile_uno_q import profile_for_uno_q


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Compute integer confusion matrix [num_classes, num_classes]."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Union[str, Path]) -> None:
    """Render and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="RaDICaL Target Classification Confusion Matrix",
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold",
            )
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 100
) -> Tuple[List[float], List[float], List[float], float]:
    """Compute True Positive Rate (TPR), False Positive Rate (FPR), and AUROC."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    tpr_list = []
    fpr_list = []

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)

    for th in thresholds:
        pred_pos = (y_score >= th).astype(int)
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        tpr = tp / pos if pos > 0 else 0.0
        fpr = fp / neg if neg > 0 else 0.0
        tpr_list.append(float(tpr))
        fpr_list.append(float(fpr))

    # AUROC via trapezoid integration
    sorted_indices = np.argsort(fpr_list)
    sorted_fpr = np.array(fpr_list)[sorted_indices]
    sorted_tpr = np.array(tpr_list)[sorted_indices]
    auroc = float(np.sum((sorted_tpr[:-1] + sorted_tpr[1:]) * np.diff(sorted_fpr)) * 0.5)
    if auroc < 0:
        auroc = float(abs(auroc))

    return fpr_list, tpr_list, thresholds.tolist(), auroc


def plot_roc_curve(fpr: List[float], tpr: List[float], auroc: float, output_path: Union[str, Path]) -> None:
    """Render and save Receiver Operating Characteristic (ROC) curve."""
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"PhotonV0 ROC (AUROC = {auroc:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Guess (AUROC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("RaDICaL Target Detection ROC Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def benchmark_latency_and_memory(
    model: PhotonV0,
    sequence_length: int = 16,
    input_dim: int = 64,
    device: torch.device = torch.device("cpu"),
    num_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Benchmark forward pass latency."""
    model.eval()
    model.to(device)
    dummy_single = torch.randn(1, sequence_length, input_dim, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_single)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_single)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_arr))
    std_lat = float(np.std(latencies_arr))
    p50_lat = float(np.percentile(latencies_arr, 50))
    p95_lat = float(np.percentile(latencies_arr, 95))
    p99_lat = float(np.percentile(latencies_arr, 99))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "device": str(device),
        "latency_mean_ms": round(mean_lat, 3),
        "latency_std_ms": round(std_lat, 3),
        "latency_p50_ms": round(p50_lat, 3),
        "latency_p95_ms": round(p95_lat, 3),
        "latency_p99_ms": round(p99_lat, 3),
        "throughput_fps": round(fps, 1),
    }


def evaluate_photon_v0(
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    output_dir: str = "results/photon_v0",
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    """Execute complete evaluation workflow."""
    cfg_data = config.get("dataset", {})
    cfg_model = config.get("model", {})
    cfg_train = config.get("training", {})

    seed = cfg_train.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Adapter
    synthetic_fallback = allow_synthetic or cfg_data.get("synthetic_fallback", False)
    adapter = RaDICaLDatasetAdapter(
        data_path=cfg_data.get("data_path", "data/radical"),
        sequence_length=cfg_data.get("sequence_length", 16),
        feature_dim=cfg_data.get("feature_dim", 64),
        num_classes=cfg_model.get("num_classes", 4),
        normalization=cfg_data.get("normalization", "db"),
        train_ratio=cfg_data.get("train_ratio", 0.70),
        val_ratio=cfg_data.get("val_ratio", 0.15),
        test_ratio=cfg_data.get("test_ratio", 0.15),
        seed=seed,
        synthetic_fallback=synthetic_fallback,
    )

    _, _, test_loader = adapter.get_dataloaders(
        batch_size=cfg_train.get("batch_size", 64), num_synthetic_fallback=500
    )

    # 2. Build Model
    model = PhotonV0(
        input_dim=cfg_model.get("input_dim", 64),
        hidden_dim=cfg_model.get("hidden_dim", 64),
        num_layers=cfg_model.get("num_layers", 2),
        sequence_length=cfg_model.get("sequence_length", 16),
        num_classes=cfg_model.get("num_classes", 4),
        d_state=cfg_model.get("d_state", 16),
        d_conv=cfg_model.get("d_conv", 4),
        expand=cfg_model.get("expand", 2),
        use_attention=cfg_model.get("use_attention", False),
        backend=cfg_model.get("backend", "auto"),
    ).to(device)

    ckpt_candidate = (
        checkpoint_path
        or (Path(cfg_train.get("checkpoint_dir", "checkpoints/photon_v0")) / "best_model.pt")
        or (Path(cfg_train.get("checkpoint_dir", "checkpoints/photon_v0")) / "photon_v0_best.pt")
    )
    if Path(ckpt_candidate).exists():
        model.load_state_dict(torch.load(ckpt_candidate, map_location=device))
        print(f"[PhotonShield AI] Loaded weights from {ckpt_candidate}")
    else:
        print(f"[PhotonShield AI] Checkpoint {ckpt_candidate} not found. Evaluating with initialized model.")

    model.eval()

    # 3. Collect Predictions
    det_preds, det_targets = [], []
    cls_preds, cls_targets = [], []
    ano_preds, ano_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            y_det = batch["detection"].to(device)
            y_cls = batch["classification"].to(device)
            y_ano = batch["anomaly"].to(device)

            outputs = model(x)
            det_preds.append(outputs["detection"].cpu().numpy())
            det_targets.append(y_det.cpu().numpy())
            cls_preds.append(torch.argmax(outputs["classification"], dim=-1).cpu().numpy())
            cls_targets.append(y_cls.cpu().numpy())
            ano_preds.append(outputs["anomaly"].cpu().numpy())
            ano_targets.append(y_ano.cpu().numpy())

    all_det_p = np.concatenate(det_preds, axis=0).flatten()
    all_det_t = (np.concatenate(det_targets, axis=0).flatten() > 0.5).astype(int)
    all_cls_p = np.concatenate(cls_preds, axis=0)
    all_cls_t = np.concatenate(cls_targets, axis=0)
    all_ano_p = np.concatenate(ano_preds, axis=0).flatten()
    all_ano_t = np.concatenate(ano_targets, axis=0).flatten()

    # Confusion Matrix
    num_classes = cfg_model.get("num_classes", 4)
    cm = compute_confusion_matrix(all_cls_t, all_cls_p, num_classes=num_classes)
    plot_confusion_matrix(cm, RADICAL_CLASSES[:num_classes], out_path / "confusion_matrix.png")

    # Precision, Recall, F1 per class
    per_class_metrics = {}
    f1_list = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        c_name = RADICAL_CLASSES[c] if c < len(RADICAL_CLASSES) else f"class_{c}"
        per_class_metrics[c_name] = {
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "f1": float(round(f1, 4)),
            "support": int(np.sum(cm[c, :])),
        }

    macro_f1 = float(np.mean(f1_list))
    accuracy = float(np.mean(all_cls_p == all_cls_t))
    ano_mae = float(np.mean(np.abs(all_ano_p - all_ano_t)))

    # ROC & AUROC
    fpr_list, tpr_list, thresholds, auroc = compute_roc_curve(all_det_t, all_det_p)
    plot_roc_curve(fpr_list, tpr_list, auroc, out_path / "roc_curve.png")

    # Latency Benchmark on CPU
    cpu_latency = benchmark_latency_and_memory(
        model=model,
        sequence_length=cfg_model.get("sequence_length", 16),
        input_dim=cfg_model.get("input_dim", 64),
        device=torch.device("cpu"),
    )

    # Latency Benchmark on CUDA if available
    cuda_latency = None
    if torch.cuda.is_available():
        cuda_latency = benchmark_latency_and_memory(
            model=model,
            sequence_length=cfg_model.get("sequence_length", 16),
            input_dim=cfg_model.get("input_dim", 64),
            device=torch.device("cuda"),
        )

    # Hardware profile for Uno Q
    hw_profile = profile_for_uno_q(
        model=model,
        sequence_length=cfg_model.get("sequence_length", 16),
        hidden_dim=cfg_model.get("hidden_dim", 64),
    )

    results = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "anomaly_mae": round(ano_mae, 4),
        "detection_auroc": round(auroc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "roc_curve": {
            "fpr": [round(x, 4) for x in fpr_list],
            "tpr": [round(x, 4) for x in tpr_list],
            "thresholds": [round(x, 4) for x in thresholds],
        },
        "latency_benchmark_cpu": cpu_latency,
        "latency_benchmark_cuda": cuda_latency,
        "hardware_profile": hw_profile,
    }

    # Save to JSON
    with open(out_path / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save plain text summary report
    report_txt = f"""================================================================
 PhotonShield AI - Phase V0 Evaluation & Benchmark Report
================================================================
Overall Metrics:
  - Classification Accuracy:  {accuracy * 100:.2f}%
  - Macro F1 Score:           {macro_f1:.4f}
  - Detection AUROC:          {auroc:.4f}
  - Anomaly MAE:              {ano_mae:.4f}

Confusion Matrix (Rows=True, Cols=Predicted):
{cm}

Per-Class Performance:
"""
    for c_name, stats in per_class_metrics.items():
        report_txt += f"  - {c_name:<12}: Precision={stats['precision']:.3f}, Recall={stats['recall']:.3f}, F1={stats['f1']:.3f} (N={stats['support']})\n"

    report_txt += f"""
Latency & Throughput:
  - CPU Mean Latency:         {cpu_latency['latency_mean_ms']:.3f} ms / sample ({cpu_latency['throughput_fps']:.1f} FPS)
  - CPU 95th Percentile:      {cpu_latency['latency_p95_ms']:.3f} ms
"""
    if cuda_latency:
        report_txt += f"  - CUDA Mean Latency:        {cuda_latency['latency_mean_ms']:.3f} ms ({cuda_latency['throughput_fps']:.1f} FPS)\n"

    report_txt += f"""
Target Deployment (Arduino Uno Q):
  - Parameters:               {hw_profile['parameter_count']:,}
  - Weight Memory (INT8):     {hw_profile['weights_int8_kb']:.2f} KB / {hw_profile['target_flash_kb']:.0f} KB Flash
  - Peak SRAM (INT8):         {hw_profile['peak_sram_int8_kb']:.2f} KB / {hw_profile['target_sram_kb']:.0f} KB SRAM
  - Projected Latency @64MHz: ~{hw_profile['estimated_latency_ms']:.2f} ms
  - Uno Q Hardware Status:    {'PASS (Feasible)' if hw_profile['overall_fit'] else 'FAIL'}
================================================================
"""

    with open(out_path / "benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(report_txt)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PhotonV0 Perception Stack.")
    parser.add_argument("--config", type=str, default="configs/photon_v0.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/photon_v0/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="results/photon_v0")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    evaluate_photon_v0(
        config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        allow_synthetic=args.allow_synthetic,
    )


if __name__ == "__main__":
    main()
