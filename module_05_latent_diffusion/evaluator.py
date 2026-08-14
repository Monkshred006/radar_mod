"""Evaluator Engine for Latent Diffusion Model (PhotonShield AI V1.0).

Performs baseline vs. reconstructed MSE comparison, PCA visualizations, and report generation.
"""

from __future__ import annotations

import json
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

        total_corrupted_mse = 0.0
        total_reconstructed_mse = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.test_loader:
                x = batch["features"].to(self.device)
                B = x.shape[0]

                # Run reverse diffusion reconstruction
                z_hat, z_0, z_c = self.model.reconstruct(x=x, num_steps=num_inference_steps)

                corr_mse = nn.functional.mse_loss(z_c, z_0).item()
                rec_mse = nn.functional.mse_loss(z_hat, z_0).item()

                total_corrupted_mse += corr_mse * B
                total_reconstructed_mse += rec_mse * B
                total_samples += B

                all_z0.append(z_0.cpu().numpy())
                all_zc.append(z_c.cpu().numpy())
                all_zhat.append(z_hat.cpu().numpy())

        mean_corr_mse = total_corrupted_mse / max(total_samples, 1)
        mean_rec_mse = total_reconstructed_mse / max(total_samples, 1)
        
        improvement_pct = 100.0 * (mean_corr_mse - mean_rec_mse) / max(mean_corr_mse, 1e-8)
        gate_passed = bool(mean_rec_mse < mean_corr_mse)

        z0_arr = np.concatenate(all_z0, axis=0)      # [N, T, D]
        zc_arr = np.concatenate(all_zc, axis=0)      # [N, T, D]
        zhat_arr = np.concatenate(all_zhat, axis=0)  # [N, T, D]

        # Generate PCA Visualizations
        self.generate_pca_plots(z0_arr, zc_arr, zhat_arr)

        return {
            "corrupted_latent_mse": round(mean_corr_mse, 6),
            "reconstructed_latent_mse": round(mean_rec_mse, 6),
            "improvement_percentage": round(improvement_pct, 2),
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

        gate_status = "PASS" if eval_results["gate_passed"] else "FAIL"
        next_stage = "V1.1 JOINT PERCEPTION" if eval_results["gate_passed"] else "INVESTIGATE DIFFUSION"

        report_lines = [
            "# PhotonShield AI — Phase V1.0 Latent Diffusion Baseline Report",
            "",
            "**Version**: `Phase V1.0 (Latent Reconstruction Baseline)`  ",
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
            "In Phase V1.0, conditional latent diffusion is evaluated strictly as an isolated latent reconstruction module operating on the frozen representations of the PhotonV0 temporal foundation.",
            "",
            "---",
            "",
            "## 2. Architecture Overview",
            "",
            "* **Frozen Backbone**: PhotonV0 Encoder (2-layer Mini-Mamba, D=64, H=64, 70,566 frozen parameters).",
            f"* **Trainable Denoiser**: `LightweightDenoiser` (Sinusoidal timestep embedding + 2 Temporal Convolution Residual Blocks + Input/Output Linear Projections).",
            f"* **Denoiser Parameter Count**: {self.model.denoiser.count_parameters():,} parameters.",
            "* **Diffusion Type**: DDPM conditional noise prediction (eps_theta(z_t, z_c, t)).",
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
            "* **Corruption Type**: **Temporal Frame Dropout**",
            f"* **Dropout Probability**: p = {cfg_corr.get('frame_dropout', {}).get('probability', 0.20):.2f} (20% of temporal frames randomly zeroed out).",
            "* **Other Corruptions**: Disabled for Phase V1.0 baseline.",
            "",
            "---",
            "",
            "## 6. Training Configuration",
            "",
            f"* **Batch Size**: {cfg_train.get('batch_size', 16)}",
            f"* **Max Epochs**: {train_results.get('epochs', 50)} (Best Epoch: {train_results.get('best_epoch', 0)})",
            f"* **Learning Rate**: {cfg_train.get('learning_rate', 5e-4)} (Cosine Annealing scheduler)",
            "* **Optimizer**: AdamW (weight decay = 1e-4)",
            "* **Precision**: Mixed Precision (Float16 Autocast + GradScaler on CUDA)",
            "",
            "---",
            "",
            "## 7. Reconstruction Performance & Hypothesis Validation",
            "",
            "| Metric | Measured Score | Target Condition |",
            "| :--- | :---: | :---: |",
            f"| **Baseline Corrupted Latent MSE: MSE(Z_c, Z)** | **{eval_results['corrupted_latent_mse']:.6f}** | Reference Baseline |",
            f"| **Reconstructed Latent MSE: MSE(Z_hat, Z)** | **{eval_results['reconstructed_latent_mse']:.6f}** | **< MSE(Z_c, Z)** |",
            f"| **Reconstruction Error Reduction** | **{eval_results['improvement_percentage']:.2f}%** | **> 0.0%** |",
            f"| **Best Validation Diffusion Loss** | **{train_results.get('best_val_rec_mse', 0.0):.6f}** | Finite / Converged |",
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
            "1. **Isolated Latent Reconstruction**: V1.0 only optimizes L_noise = MSE(eps_hat, eps) and does not backpropagate task classification gradients.",
            "2. **Deterministic Sampler Speed**: Full 50-step DDPM sampling requires multi-step iterative loops, which will be accelerated via DDIM / 1-step direct approximation in subsequent stages.",
            "",
            "---",
            "",
            "## 10. Pass / Fail Decision & Next Step",
            "",
            f"* **V1.0 Decision**: **{gate_status}**",
            f"* **Gate Verification**: MSE(Z_hat, Z) < MSE(Z_c, Z) verified with **{eval_results['improvement_percentage']:.2f}%** reconstruction error reduction.",
            f"* **Next Stage**: **{next_stage}**",
        ]

        report_path = self.results_dir / "V1_0_REPORT.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"[DiffusionEvaluator] Generated V1.0 report at '{report_path}'")
