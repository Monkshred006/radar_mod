"""Run reverse trajectory diagnostics and generate visualization artifacts for Phase V1.0."""

from __future__ import annotations
import sys
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter
from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_05_latent_diffusion.latent_diffusion import LatentDiffusionModel
from module_05_latent_diffusion.losses import DiffusionLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running diagnostics on {device}...")

    # Load frozen V0
    v0_path = REPO_ROOT / "checkpoints" / "v0_frozen" / "best_model.pt"
    encoder = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16, num_classes=4, use_attention=False).to(device)
    encoder.load_state_dict(torch.load(v0_path, map_location=device))
    encoder.eval()

    # Load V1 diffusion model with best checkpoint
    config_path = REPO_ROOT / "configs" / "photon_v1_diffusion.yaml"
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = LatentDiffusionModel(
        v0_checkpoint_path=v0_path,
        latent_dim=config.get("diffusion", {}).get("latent_dim", 64),
        hidden_dim=config.get("diffusion", {}).get("hidden_dim", 128),
        num_blocks=config.get("diffusion", {}).get("num_blocks", 2),
        timesteps=config.get("diffusion", {}).get("timesteps", 50),
        beta_schedule=config.get("diffusion", {}).get("beta_schedule", "linear"),
        corruption_config=config.get("corruption"),
        loss_config=config.get("losses"),
    ).to(device)
    ckpt_path = REPO_ROOT / "checkpoints" / "v1_diffusion" / "best_diffusion.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.denoiser.load_state_dict(ckpt.get("denoiser_state_dict", ckpt))
        print("Loaded best denoiser weights from checkpoint.")
    model.eval()

    # Get validation dataloader
    adapter = RaDICaLDatasetAdapter(
        data_path="C:/Users/worka/research/photonpinn/data/radical",
        splits_dir="C:/Users/worka/research/photonpinn/data/radical/splits",
        sequence_length=16,
        feature_dim=64,
        num_classes=4,
        normalization="db",
        seed=42,
        synthetic_fallback=False,
    )
    _, val_loader, test_loader = adapter.get_dataloaders(batch_size=16)

    # 1. Deterministic Reproducibility Check
    print("\n--- STEP 7: DETERMINISTIC REPRODUCIBILITY CHECK ---")
    val_batch = next(iter(val_loader))
    x_sample = val_batch["features"].to(device)
    z0_sample = model.encode(x_sample)
    zc_sample, mask_sample = model.corruption(z0_sample)

    out1_det = model.scheduler.reconstruct(model.denoiser, zc_sample, mask_sample, num_inference_steps=50, deterministic=True)
    out2_det = model.scheduler.reconstruct(model.denoiser, zc_sample, mask_sample, num_inference_steps=50, deterministic=True)
    max_diff_det = float(torch.max(torch.abs(out1_det - out2_det)).item())
    print(f"Deterministic Run 1 vs Run 2 Max Absolute Difference: {max_diff_det:.10e}")
    assert max_diff_det < 1e-7, f"Deterministic check failed: max_diff={max_diff_det}"

    out1_stoch = model.scheduler.reconstruct(model.denoiser, zc_sample, mask_sample, num_inference_steps=50, deterministic=False)
    out2_stoch = model.scheduler.reconstruct(model.denoiser, zc_sample, mask_sample, num_inference_steps=50, deterministic=False)
    max_diff_stoch = float(torch.max(torch.abs(out1_stoch - out2_stoch)).item())
    print(f"Stochastic Run 1 vs Run 2 Max Absolute Difference: {max_diff_stoch:.6f}")
    assert max_diff_stoch > 0.0, "Stochastic check failed (no variation)"

    # 2. Observed-Frame Preservation
    print("\n--- STEP 8: OBSERVED-FRAME PRESERVATION CHECK ---")
    obs_metrics = DiffusionLoss.reconstruction_metrics(out1_det, z0_sample, mask_sample)
    print(f"Observed-Frame MSE: {obs_metrics['observed_mse']:.10e}")
    print(f"Missing-Frame MSE : {obs_metrics['missing_mse']:.6f}")
    print(f"Full-Sequence MSE : {obs_metrics['full_mse']:.6f}")

    # 3. Reverse-Diffusion Trajectory Diagnostics (Step 9)
    print("\n--- STEP 9: REVERSE-DIFFUSION TRAJECTORY DIAGNOSTICS ---")
    # Take sample 0 [1, 16, 64]
    x_single = x_sample[:1]
    z0_single = z0_sample[:1]
    zc_single = zc_sample[:1]
    mask_single = mask_sample[:1]

    # Trace timesteps 50, 40, 30, 20, 10, 0
    timesteps = list(reversed(range(50)))
    B, T, D = zc_single.shape
    sqrt_alpha_start = model.scheduler.sqrt_alphas_cumprod[49]
    z_t = (mask_single * (sqrt_alpha_start * zc_single)) + ((1.0 - mask_single) * torch.zeros_like(zc_single))

    trajectory_stats = {}
    target_checkpoints = [50, 40, 30, 20, 10, 0]

    # At t=50 (before step 49)
    missing_mask = 1.0 - mask_single
    diff_50 = (z_t - z0_single) * missing_mask
    mse_50 = float(((diff_50 ** 2).sum() / torch.clamp(missing_mask.sum() * 64, min=1.0)).item())
    trajectory_stats["t=50"] = {
        "mean": float(z_t.mean().item()),
        "std": float(z_t.std().item()),
        "min": float(z_t.min().item()),
        "max": float(z_t.max().item()),
        "missing_frame_mse": mse_50,
    }

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = model.denoiser(z_t=z_t, condition=zc_single, timestep=t_tensor, mask=mask_single)
        
        alpha_bar_t = model.scheduler.alphas_cumprod[t]
        sqrt_alpha_t = model.scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_t = model.scheduler.sqrt_one_minus_alphas_cumprod[t]

        pred_z0 = (z_t - (sqrt_one_minus_t * eps_pred)) / torch.clamp(sqrt_alpha_t, min=1e-6)

        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            sqrt_alpha_prev = model.scheduler.sqrt_alphas_cumprod[t_prev]
            sqrt_one_minus_prev = model.scheduler.sqrt_one_minus_alphas_cumprod[t_prev]
            z_prev_gen = (sqrt_alpha_prev * pred_z0) + (sqrt_one_minus_prev * eps_pred)
            z_prev_known = sqrt_alpha_prev * zc_single
            z_t = (mask_single * z_prev_known) + ((1.0 - mask_single) * z_prev_gen)
            current_t = t_prev + 1
        else:
            z_t = (mask_single * zc_single) + ((1.0 - mask_single) * pred_z0)
            current_t = 0

        if current_t in target_checkpoints:
            diff_t = (z_t - z0_single) * missing_mask
            mse_t = float(((diff_t ** 2).sum() / torch.clamp(missing_mask.sum() * 64, min=1.0)).item())
            trajectory_stats[f"t={current_t}"] = {
                "mean": float(z_t.mean().item()),
                "std": float(z_t.std().item()),
                "min": float(z_t.min().item()),
                "max": float(z_t.max().item()),
                "missing_frame_mse": mse_t,
            }

    results_dir = REPO_ROOT / "results" / "photon_v1"
    results_dir.mkdir(parents=True, exist_ok=True)
    diag_path = results_dir / "reverse_trajectory_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(trajectory_stats, f, indent=2)
    print(f"Saved reverse trajectory diagnostics to: {diag_path}")
    print(json.dumps(trajectory_stats, indent=2))

    # 4. Generate Visualizations (Step 10)
    print("\n--- STEP 10: GENERATING VISUALIZATIONS ---")
    pca = PCA(n_components=2)
    z0_np = z0_single[0].detach().cpu().numpy()     # [16, 64]
    zc_np = zc_single[0].detach().cpu().numpy()     # [16, 64]
    zhat_np = z_t[0].detach().cpu().numpy()         # [16, 64]
    mask_np = mask_single[0].detach().cpu().numpy() # [16, 1]

    # Fit PCA on clean sequence
    z0_pca = pca.fit_transform(z0_np)
    zc_pca = pca.transform(zc_np)
    zhat_pca = pca.transform(zhat_np)

    time_steps = np.arange(16)

    # 1. Clean Latent Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(z0_pca[:, 0], z0_pca[:, 1], c=time_steps, cmap="viridis", s=80, edgecolors="k")
    ax.plot(z0_pca[:, 0], z0_pca[:, 1], color="#2ca02c", alpha=0.6, lw=1.5)
    for t_idx, (px, py) in enumerate(z0_pca):
        ax.text(px + 0.05, py + 0.05, f"t={t_idx}", fontsize=8)
    plt.colorbar(scatter, ax=ax, label="Temporal Frame Index")
    ax.set_title("Clean Latent Trajectory (PCA)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(results_dir / "clean_latent.png", dpi=200)
    plt.close()

    # 2. Corrupted Latent Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    obs_idx = np.where(mask_np[:, 0] == 1.0)[0]
    miss_idx = np.where(mask_np[:, 0] == 0.0)[0]
    ax.scatter(zc_pca[obs_idx, 0], zc_pca[obs_idx, 1], color="#1f77b4", s=80, label="Observed", edgecolors="k")
    ax.scatter(zc_pca[miss_idx, 0], zc_pca[miss_idx, 1], color="#d62728", marker="x", s=100, label="Dropped (Zeroed)", lw=2)
    ax.set_title("Corrupted Latent State (p=0.20 Dropout)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(results_dir / "corrupted_latent.png", dpi=200)
    plt.close()

    # 3. Reconstructed Latent Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(zhat_pca[:, 0], zhat_pca[:, 1], c=time_steps, cmap="plasma", s=80, edgecolors="k")
    ax.plot(zhat_pca[:, 0], zhat_pca[:, 1], color="#9467bd", alpha=0.6, lw=1.5)
    for t_idx, (px, py) in enumerate(zhat_pca):
        ax.text(px + 0.05, py + 0.05, f"t={t_idx}", fontsize=8)
    plt.colorbar(scatter, ax=ax, label="Temporal Frame Index")
    ax.set_title("Reconstructed Latent Trajectory (DDIM)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(results_dir / "reconstructed_latent.png", dpi=200)
    plt.close()

    # 4. Reconstruction Comparison Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(z0_pca[:, 0], z0_pca[:, 1], "o-", color="#2ca02c", label="Ground Truth z0", lw=2, markersize=8)
    ax.plot(zhat_pca[:, 0], zhat_pca[:, 1], "s--", color="#9467bd", label="Reconstructed z_hat", lw=2, markersize=7)
    ax.scatter(zc_pca[miss_idx, 0], zc_pca[miss_idx, 1], color="#d62728", marker="x", s=120, label="Missing Frames (Dropout)", lw=3, zorder=5)
    for idx in miss_idx:
        ax.annotate(
            f"Frame {idx}",
            xy=(zhat_pca[idx, 0], zhat_pca[idx, 1]),
            xytext=(zhat_pca[idx, 0] + 0.2, zhat_pca[idx, 1] + 0.2),
            arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.5),
            fontweight="bold",
            fontsize=9,
        )
    ax.set_title("PhotonShield V1.0 Latent Temporal Inpainting Comparison", fontweight="bold", fontsize=12)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(results_dir / "reconstruction_comparison.png", dpi=200)
    plt.close()

    print("Generated all 4 visualization artifacts in results/photon_v1/")


if __name__ == "__main__":
    main()
