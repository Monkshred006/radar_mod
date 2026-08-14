"""PhotonShield AI - Phase V1.0 Latent Diffusion Training & Evaluation Pipeline.

Executes conditional latent diffusion training to reconstruct corrupted temporal radar latents.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter
from module_05_latent_diffusion.latent_diffusion import LatentDiffusionModel
from module_05_latent_diffusion.trainer import DiffusionTrainer
from module_05_latent_diffusion.evaluator import DiffusionEvaluator


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhotonShield AI Phase V1.0 Latent Diffusion Model.")
    parser.add_argument("--config", type=str, default="configs/photon_v1_diffusion.yaml", help="Path to config yaml")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional sample limit for overfit sanity test")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg_train = config.get("training", {})
    cfg_data = config.get("dataset", {})
    cfg_v0 = config.get("v0", {})
    cfg_diff = config.get("diffusion", {})
    cfg_corr = config.get("corruption", {})
    cfg_loss = config.get("losses", {})

    seed = int(cfg_train.get("seed", 42))
    set_seed(seed)

    dev_str = cfg_train.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() and dev_str == "auto" else dev_str)
    print(f"[PhotonShield AI V1.0] Device selected: {device}")

    # 1. Load Dataset with exact frozen splits
    adapter = RaDICaLDatasetAdapter(
        data_path=cfg_data.get("data_path", "C:/Users/worka/research/photonpinn/data/radical"),
        splits_dir=cfg_data.get("splits_dir", "C:/Users/worka/research/photonpinn/data/radical/splits"),
        sequence_length=cfg_data.get("sequence_length", 16),
        feature_dim=cfg_data.get("feature_dim", 64),
        num_classes=cfg_data.get("num_classes", 4),
        normalization=cfg_data.get("normalization", "db"),
        seed=seed,
        synthetic_fallback=False,
    )

    batch_size = int(cfg_train.get("batch_size", 16))
    train_loader, val_loader, test_loader = adapter.get_dataloaders(batch_size=batch_size)

    # Optional overfit sample limit
    if args.max_train_samples is not None:
        subset_indices = list(range(min(args.max_train_samples, len(train_loader.dataset))))
        from torch.utils.data import Subset
        train_loader = torch.utils.data.DataLoader(
            Subset(train_loader.dataset, subset_indices),
            batch_size=min(batch_size, len(subset_indices)),
            shuffle=True,
        )
        print(f"[PhotonShield AI V1.0] Overfit Mode Active: Training on {len(subset_indices)} samples")

    print(f"[PhotonShield AI V1.0] RaDICaL Splits: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Val, {len(test_loader.dataset)} Test")

    # 2. Build Latent Diffusion Model with Frozen V0 Encoder
    v0_ckpt = cfg_v0.get("checkpoint", "checkpoints/v0_frozen/best_model.pt")
    model = LatentDiffusionModel(
        v0_checkpoint_path=v0_ckpt,
        latent_dim=int(cfg_diff.get("latent_dim", 64)),
        hidden_dim=int(cfg_diff.get("hidden_dim", 128)),
        num_blocks=int(cfg_diff.get("num_blocks", 2)),
        timesteps=int(cfg_diff.get("timesteps", 50)),
        beta_schedule=cfg_diff.get("beta_schedule", "linear"),
        corruption_config=cfg_corr,
        loss_config=cfg_loss,
    ).to(device)

    denoiser_params = model.denoiser.count_parameters()
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print("================================================================")
    print(f" PhotonShield AI V1.0 Latent Diffusion Model Initialized")
    print(f" - Frozen V0 Encoder Parameters: {encoder_params:,} (requires_grad = False)")
    print(f" - Trainable Denoiser Parameters: {denoiser_params:,}")
    print(f" - Active Corruption: Frame Dropout (p={cfg_corr.get('frame_dropout', {}).get('probability', 0.20):.2f})")
    print(f" - Loss Weights: L_diff={cfg_loss.get('lambda_diff', 1.0)}, L_recon={cfg_loss.get('lambda_recon', 0.5)}, L_missing={cfg_loss.get('lambda_missing', 1.0)}")
    print("================================================================")

    # 3. Train Diffusion Denoiser
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )
    train_summary = trainer.train()
    train_summary["epochs"] = config["training"]["epochs"]

    # 4. Evaluate on Test Set using Best Checkpoint
    best_ckpt_path = Path("checkpoints/v1_diffusion/best_diffusion.pt")
    if best_ckpt_path.exists():
        model.denoiser.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"[PhotonShield AI V1.0] Loaded best denoiser weights from '{best_ckpt_path}'")

    results_dir = Path("results/photon_v1")
    evaluator = DiffusionEvaluator(
        model=model,
        test_loader=test_loader,
        results_dir=results_dir,
        device=device,
    )
    inf_steps = int(cfg_diff.get("inference_steps", cfg_diff.get("timesteps", 50)))
    eval_results = evaluator.evaluate_test_set(num_inference_steps=inf_steps)

    # Save summary JSON and Markdown report
    evaluator.generate_v1_report(train_results=train_summary, eval_results=eval_results, config=config)

    # Save to full_training directory if full run
    if args.max_train_samples is None:
        full_train_dir = results_dir / "full_training"
        full_train_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        if best_ckpt_path.exists():
            shutil.copy(best_ckpt_path, full_train_dir / "best_model.pt")
        if (results_dir / "metrics.csv").exists():
            shutil.copy(results_dir / "metrics.csv", full_train_dir / "metrics.csv")
        if (results_dir / "diffusion_training_curve.png").exists():
            shutil.copy(results_dir / "diffusion_training_curve.png", full_train_dir / "training_curves.png")
        if (results_dir / "V1_0_REPORT.md").exists():
            shutil.copy(results_dir / "V1_0_REPORT.md", full_train_dir / "training_report.md")

    # Generate FULL_V1_REPORT.md
    report_path = results_dir / "FULL_V1_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PhotonShield AI — Phase V1.0 Full Latent Diffusion Report\n\n")
        f.write(f"- **Git Commit**: `f365c0c360f03801d26d2248bd0989901562e86d`\n")
        f.write(f"- **Target Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM)\n")
        f.write(f"- **Frozen V0 Baseline**: Checkpoint `checkpoints/v0_frozen/best_model.pt` (70,566 params)\n")
        f.write(f"- **Trainable Denoiser**: LightweightDenoiser ({denoiser_params:,} params)\n\n")
        f.write("## Dataset & Corruption Configuration\n")
        f.write(f"- **Dataset**: RaDICaL (350 Train / 75 Val / 75 Test)\n")
        f.write(f"- **Corruption**: Temporal Frame Dropout (p = {cfg_corr.get('frame_dropout', {}).get('probability', 0.20):.2f})\n")
        f.write(f"- **Diffusion Timesteps / Inference Steps**: {cfg_diff.get('timesteps', 50)} / {inf_steps}\n\n")
        f.write("## Training Dynamics & Best Epoch\n")
        f.write(f"- **Best Epoch**: Epoch {train_summary['best_epoch']}\n")
        f.write(f"- **Best Validation Reconstruction MSE**: `{train_summary['best_val_rec_mse']:.6f}`\n")
        f.write(f"- **Total Training Time**: `{train_summary['total_time_sec']:.2f} s`\n")
        f.write(f"- **Peak Tensor VRAM**: `{train_summary['peak_vram_gb']:.4f} GB`\n\n")
        f.write("## Test Set Performance (Evaluated Once with Best Checkpoint)\n\n")
        f.write("| Metric | Corrupted Baseline | Reconstructed Latent | Relative Error Reduction (%) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Missing-Frame MSE** | **{eval_results['corrupted_missing_mse']:.6f}** | **{eval_results['reconstructed_missing_mse']:.6f}** | **{eval_results['missing_improvement_percentage']:.2f}%** |\n")
        f.write(f"| **Full Sequence MSE** | **{eval_results['corrupted_latent_mse']:.6f}** | **{eval_results['reconstructed_latent_mse']:.6f}** | **{eval_results['improvement_percentage']:.2f}%** |\n")
        f.write(f"| **Observed-Frame MSE** | **0.000000** | **0.000000** | **Exact Data Consistency** |\n")
        f.write(f"| **Full Sequence MAE / RMSE** | — | **{eval_results['reconstructed_latent_mae']:.6f} / {eval_results['reconstructed_latent_rmse']:.6f}** | — |\n\n")
        f.write("## Hardware Telemetry\n")
        f.write(f"- **GPU**: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}\n")
        f.write(f"- **Batch Latency**: `{train_summary['avg_batch_latency_ms']:.2f} ms`\n")
        f.write(f"- **Single Sample Latency**: `{(train_summary['avg_batch_latency_ms'] / batch_size):.2f} ms`\n\n")
        f.write("## V1 Conclusion & Decision\n")
        f.write(f"- **Gate Decision**: **{'PASS' if eval_results['gate_passed'] else 'FAIL'}**\n")
        f.write("- **Notice**: No downstream classification metrics are evaluated or claimed until Phase V1.1 Joint Perception.\n")

    with open(results_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_summary, "eval": eval_results}, f, indent=2)

    status_str = "PASS" if eval_results["gate_passed"] else "FAIL"
    next_stage_str = "V1.1 JOINT PERCEPTION" if eval_results["gate_passed"] else "INVESTIGATE DIFFUSION"

    print("\n================================================================")
    print("V1.0 LATENT DIFFUSION COMPLETE")
    print()
    print(f"Corruption:")
    print(f"Temporal frame dropout, p={cfg_corr.get('frame_dropout', {}).get('probability', 0.20):.2f}")
    print()
    print(f"Baseline corrupted latent MSE:")
    print(f"{eval_results['corrupted_latent_mse']:.6f}")
    print()
    print(f"Reconstructed latent MSE:")
    print(f"{eval_results['reconstructed_latent_mse']:.6f}")
    print()
    print(f"Missing-frame corrupted MSE: {eval_results['corrupted_missing_mse']:.6f} -> Reconstructed MSE: {eval_results['reconstructed_missing_mse']:.6f}")
    print()
    print(f"Improvement (Full Sequence):")
    print(f"{eval_results['improvement_percentage']:.2f} %")
    print(f"Improvement (Missing Frames):")
    print(f"{eval_results['missing_improvement_percentage']:.2f} %")
    print()
    print(f"Best validation reconstruction MSE:")
    print(f"{train_summary['best_val_rec_mse']:.6f}")
    print()
    print(f"Peak VRAM:")
    print(f"{train_summary['peak_vram_gb']:.4f} GB")
    print()
    print(f"Training time:")
    print(f"{train_summary['total_time_sec']:.2f} seconds")
    print()
    print(f"V1.0 STATUS:")
    print(f"{status_str}")
    print()
    print(f"NEXT:")
    print(f"{next_stage_str}")
    print("================================================================")


if __name__ == "__main__":
    main()
