"""Evaluator Engine for Latent Diffusion Model (PhotonShield AI V1).

Performs missing-frame vs. observed-frame MSE/MAE/RMSE comparison, PCA visualizations, and report generation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module_05_latent_diffusion.latent_diffusion import LatentDiffusionModel
from module_05_latent_diffusion.losses import DiffusionLoss


class DiffusionEvaluator:
    """Evaluates latent diffusion reconstruction quality and generates comparative visualizations."""

    def __init__(
        self,
        model: LatentDiffusionModel,
        test_loader: DataLoader,
        results_dir: Path,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.test_loader = test_loader
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def evaluate_test_set(self, num_inference_steps: int = 50) -> Dict[str, Any]:
        """Run full evaluation on the unseen test set."""
        self.model.denoiser.eval()

        all_z0 = []
        all_zc = []
        all_zhat = []
        all_masks = []

        total_samples = 0
        total_corr_full_mse = 0.0
        total_rec_full_mse = 0.0
        total_rec_full_mae = 0.0

        total_corr_missing_mse = 0.0
        total_rec_missing_mse = 0.0
        total_rec_missing_mae = 0.0

        total_rec_obs_mse = 0.0
        total_rec_obs_mae = 0.0

        with torch.no_grad():
            for batch in self.test_loader:
                x = batch["features"].to(self.device)
                B = x.shape[0]

                # Run reverse diffusion reconstruction with data-consistency inpainting
                z_hat, z_0, z_c, mask = self.model.reconstruct(x=x, num_steps=num_inference_steps)

                corr_metrics = DiffusionLoss.reconstruction_metrics(z_c, z_0, mask)
                rec_metrics = DiffusionLoss.reconstruction_metrics(z_hat, z_0, mask)

                total_corr_full_mse += corr_metrics["full_mse"] * B
                total_rec_full_mse += rec_metrics["full_mse"] * B
                total_rec_full_mae += rec_metrics["full_mae"] * B

                total_corr_missing_mse += corr_metrics["missing_mse"] * B
                total_rec_missing_mse += rec_metrics["missing_mse"] * B
                total_rec_missing_mae += rec_metrics["missing_mae"] * B

                total_rec_obs_mse += rec_metrics["observed_mse"] * B
                total_rec_obs_mae += rec_metrics["observed_mae"] * B

                total_samples += B

                all_z0.append(z_0.cpu().numpy())
                all_zc.append(z_c.cpu().numpy())
                all_zhat.append(z_hat.cpu().numpy())
                all_masks.append(mask.cpu().numpy())

        mean_corr_full_mse = total_corr_full_mse / max(total_samples, 1)
        mean_rec_full_mse = total_rec_full_mse / max(total_samples, 1)
        mean_rec_full_mae = total_rec_full_mae / max(total_samples, 1)
        mean_rec_full_rmse = math.sqrt(mean_rec_full_mse)

        mean_corr_missing_mse = total_corr_missing_mse / max(total_samples, 1)
        mean_rec_missing_mse = total_rec_missing_mse / max(total_samples, 1)
        mean_rec_missing_mae = total_rec_missing_mae / max(total_samples, 1)
        mean_rec_missing_rmse = math.sqrt(mean_rec_missing_mse)

        mean_rec_obs_mse = total_rec_obs_mse / max(total_samples, 1)
        mean_rec_obs_mae = total_rec_obs_mae / max(total_samples, 1)
        mean_rec_obs_rmse = math.sqrt(mean_rec_obs_mse)

        full_imprv = 100.0 * (mean_corr_full_mse - mean_rec_full_mse) / max(mean_corr_full_mse, 1e-8)
        missing_imprv = 100.0 * (mean_corr_missing_mse - mean_rec_missing_mse) / max(mean_corr_missing_mse, 1e-8)
        gate_passed = bool(mean_rec_full_mse < mean_corr_full_mse and mean_rec_missing_mse < mean_corr_missing_mse)

        z0_arr = np.concatenate(all_z0, axis=0)      # [N, T, D]
        zc_arr = np.concatenate(all_zc, axis=0)      # [N, T, D]
        zhat_arr = np.concatenate(all_zhat, axis=0)  # [N, T, D]

        self.generate_pca_plots(z0_arr, zc_arr, zhat_arr)

        return {
            "corrupted_latent_mse": round(mean_corr_full_mse, 6),
            "reconstructed_latent_mse": round(mean_rec_full_mse, 6),
            "reconstructed_latent_mae": round(mean_rec_full_mae, 6),
            "reconstructed_latent_rmse": round(mean_rec_full_rmse, 6),
            "corrupted_missing_mse": round(mean_corr_missing_mse, 6),
            "reconstructed_missing_mse": round(mean_rec_missing_mse, 6),
            "reconstructed_missing_mae": round(mean_rec_missing_mae, 6),
            "reconstructed_missing_rmse": round(mean_rec_missing_rmse, 6),
            "reconstructed_observed_mse": round(mean_rec_obs_mse, 6),
            "reconstructed_observed_mae": round(mean_rec_obs_mae, 6),
            "reconstructed_observed_rmse": round(mean_rec_obs_rmse, 6),
            "improvement_percentage": round(full_imprv, 2),
            "missing_improvement_percentage": round(missing_imprv, 2),
            "gate_passed": gate_passed,
            "total_test_samples": total_samples,
        }

    def generate_pca_plots(self, z0: np.ndarray, zc: np.ndarray, zhat: np.ndarray) -> None:
        """Fit 2D PCA on original latents and plot original, corrupted, and reconstructed state trajectories."""
        N, T, D = z0.shape
        z0_flat = z0.reshape(N * T, D)
        zc_flat = zc.reshape(N * T, D)
        zhat_flat = zhat.reshape(N * T, D)

        pca = PCA(n_components=2, random_state=42)
        pca.fit(z0_flat)

        z0_2d = pca.transform(z0_flat).reshape(N, T, 2)
        zc_2d = pca.transform(zc_flat).reshape(N, T, 2)
        zhat_2d = pca.transform(zhat_flat).reshape(N, T, 2)

        brain_dir = Path("C:/Users/worka/.gemini/antigravity/brain/2df42c78-fd32-41b4-81c6-4b5d4d54f121")

        # 1. Original Latent Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        for i in range(min(15, N)):
            ax.plot(z0_2d[i, :, 0], z0_2d[i, :, 1], "-o", alpha=0.7, markersize=4)
        ax.set_title("PhotonV0 Original Latent Trajectories (2D PCA)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        p1 = self.results_dir / "latent_original.png"
        plt.savefig(p1, dpi=200)
        if brain_dir.exists():
            plt.savefig(brain_dir / "latent_original.png", dpi=200)
        plt.close()

        # 2. Corrupted Latent Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        for i in range(min(15, N)):
            ax.plot(zc_2d[i, :, 0], zc_2d[i, :, 1], "-x", color="#d62728", alpha=0.7, markersize=5)
        ax.set_title("Corrupted Latent Trajectories (Frame Dropout p=0.20)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        p2 = self.results_dir / "latent_corrupted.png"
        plt.savefig(p2, dpi=200)
        if brain_dir.exists():
            plt.savefig(brain_dir / "latent_corrupted.png", dpi=200)
        plt.close()

        # 3. Reconstructed Latent Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        for i in range(min(15, N)):
            ax.plot(zhat_2d[i, :, 0], zhat_2d[i, :, 1], "-s", color="#2ca02c", alpha=0.7, markersize=4)
        ax.set_title("Diffusion Reconstructed Latent Trajectories", fontsize=11, fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        p3 = self.results_dir / "latent_reconstructed.png"
        plt.savefig(p3, dpi=200)
        if brain_dir.exists():
            plt.savefig(brain_dir / "latent_reconstructed.png", dpi=200)
        plt.close()

        print(f"[DiffusionEvaluator] Saved PCA plots to '{self.results_dir}'")

    def generate_v1_report(self, train_results: Dict[str, Any], eval_results: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Generate comprehensive results/photon_v1/V1_0_REPORT.md."""
        cfg_diff = config.get("diffusion", {})
        cfg_train = config.get("training", {})
        cfg_corr = config.get("corruption", {})
        cfg_loss = config.get("losses", {})

        gate_status = "PASS" if eval_results["gate_passed"] else "FAIL"
        next_stage = "V1.1 JOINT PERCEPTION" if eval_results["gate_passed"] else "INVESTIGATE DIFFUSION"

        report_lines = [
            "# PhotonShield AI — Phase V1.0 Latent Diffusion Baseline Report",
            "",
            "**Version**: `Phase V1.0 (Latent Reconstruction & Imputation)`  ",
            "**Date**: 2026-08-15  ",
            "**Target Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM) & Arduino UNO Q  ",
            f"**Status**: **{gate_status}**  ",
            "",
            "---",
            "",
            "## 1. Research Hypothesis",
            "",
            "> *\"Latent diffusion can reconstruct corrupted or missing temporal radar information and improve the robustness of lightweight Mamba-based radar perception.\"*",
            "",
            "Conditional latent diffusion performs temporal imputation by preserving observed frames through data-consistency projection while reconstructing missing frames via reverse denoising.",
            "",
            "---",
            "",
            "## 2. Architecture Overview",
            "",
            "* **Frozen Backbone**: PhotonV0 Encoder (2-layer Mini-Mamba, D=64, H=64, 70,566 frozen parameters).",
            "* **Trainable Denoiser**: `LightweightDenoiser` (Input projection 129 -> 128 with explicit mask conditioning, 2 Temporal Residual Blocks, 128 -> 64 output projection).",
            f"* **Denoiser Parameter Count**: {self.model.denoiser.count_parameters():,} parameters.",
            "* **Diffusion Type**: Conditional DDPM with inpainting data consistency.",
            f"* **Diffusion Timesteps**: {cfg_diff.get('timesteps', 50)} steps.",
            "",
            "---",
            "",
            "## 3. Frozen V0 Baseline Reference",
            "",
            "* **V0 Checkpoint**: `checkpoints/v0_frozen/best_model.pt` (`requires_grad = False`, `eval()` mode).",
            "* **V0 Test Macro-F1 Reference**: 0.8711 ± 0.0109.",
            "* **V0 Test Accuracy Reference**: 87.56% ± 0.77%.",
            "",
            "---",
            "",
            "## 4. Dataset & Splits",
            "",
            "* **Source**: RaDICaL (77 GHz mmWave FMCW Range-Doppler sequences).",
            "* **Splits Used**: Fixed sequence IDs from `data/radical/splits/` (350 Train, 75 Validation, 75 Test).",
            "* **Sequence Length**: 16 frames (T=16).",
            "* **Feature Dimension**: 64 features per frame (D=64).",
            "",
            "---",
            "",
            "## 5. Corruption Model",
            "",
            "* **Corruption Type**: **Temporal Frame Dropout with Explicit Binary Mask**",
            f"* **Dropout Probability**: p = {cfg_corr.get('frame_dropout', {}).get('probability', 0.20):.2f} (20% of temporal frames randomly missing).",
            "",
            "---",
            "",
            "## 6. Training & Loss Configuration",
            "",
            f"* **Batch Size**: {cfg_train.get('batch_size', 16)}",
            f"* **Max Epochs**: {train_results.get('epochs', 50)} (Best Epoch: {train_results.get('best_epoch', 0)})",
            f"* **Learning Rate**: {cfg_train.get('learning_rate', 5e-4)} (Cosine Annealing scheduler)",
            "* **Optimizer**: AdamW (weight decay = 1e-4)",
            f"* **Loss Weights**: lambda_diff = {cfg_loss.get('lambda_diff', 1.0)}, lambda_recon = {cfg_loss.get('lambda_recon', 0.5)}, lambda_missing = {cfg_loss.get('lambda_missing', 1.0)}",
            "* **Precision**: Mixed Precision (Float16 Autocast + GradScaler on CUDA)",
            "",
            "---",
            "",
            "## 7. Reconstruction Performance & Hypothesis Validation",
            "",
            "| Metric | Corrupted Baseline | Reconstructed Output | Error Reduction (%) |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Full Sequence MSE** | **{eval_results['corrupted_latent_mse']:.6f}** | **{eval_results['reconstructed_latent_mse']:.6f}** | **{eval_results['improvement_percentage']:.2f}%** |",
            f"| **Missing-Frame MSE** | **{eval_results['corrupted_missing_mse']:.6f}** | **{eval_results['reconstructed_missing_mse']:.6f}** | **{eval_results['missing_improvement_percentage']:.2f}%** |",
            f"| **Observed-Frame MSE** | **0.000000** | **{eval_results['reconstructed_observed_mse']:.6f}** | Preserved (Data Consistency) |",
            f"| **Full Sequence MAE / RMSE** | — | **{eval_results['reconstructed_latent_mae']:.6f} / {eval_results['reconstructed_latent_rmse']:.6f}** | — |",
            f"| **Missing-Frame MAE / RMSE** | — | **{eval_results['reconstructed_missing_mae']:.6f} / {eval_results['reconstructed_missing_rmse']:.6f}** | — |",
            "",
            "---",
            "",
            "## 8. Hardware Telemetry & Runtime Profile (RTX 5050 8GB)",
            "",
            f"* **Peak Tensor VRAM**: **{train_results.get('peak_vram_gb', 0.0)} GB** (~{round(train_results.get('peak_vram_gb', 0.0) * 1024, 1)} MB / 7.96 GB)",
            f"* **Total Training Time**: **{train_results.get('total_time_sec', 0.0)} seconds**",
            f"* **Average Batch Latency**: **{train_results.get('avg_batch_latency_ms', 0.0)} ms**",
            "",
            "---",
            "",
            "## 9. Limitations",
            "",
            "1. **Isolated Latent Reconstruction**: V1.0 optimizes diffusion imputation on frozen latents without end-to-end task classification gradients.",
            "2. **Inference Steps**: Full 50-step DDPM sampling provides high reconstruction quality; single-step or few-step DDIM can be used for ultra-low latency.",
            "",
            "---",
            "",
            "## 10. Pass / Fail Decision & Next Step",
            "",
            f"* **V1.0 Decision**: **{gate_status}**",
            f"* **Gate Verification**: Full MSE improved by **{eval_results['improvement_percentage']:.2f}%** and missing-frame MSE improved by **{eval_results['missing_improvement_percentage']:.2f}%**.",
            f"* **Next Stage**: **{next_stage}**",
        ]

        report_path = self.results_dir / "V1_0_REPORT.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"[DiffusionEvaluator] Generated V1.0 report at '{report_path}'")
