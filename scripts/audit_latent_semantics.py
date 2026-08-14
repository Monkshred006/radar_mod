"""Empirical Latent Semantics Audit for PhotonShield AI Phase V2.

Performs input perturbation experiments on the frozen PhotonV0 encoder
to measure whether latent dimensions [0:30] and [30:60] preserve physical
range/Doppler feature identities, or whether features are mixed.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter
from module_04_mamba_hybrid.photon_v0 import PhotonV0


def run_semantics_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LatentSemanticsAudit] Running on: {device}")

    # 1. Load frozen PhotonV0
    v0_path = REPO_ROOT / "checkpoints" / "v0_frozen" / "best_model.pt"
    encoder = PhotonV0(
        input_dim=64,
        hidden_dim=64,
        num_layers=2,
        sequence_length=16,
        num_classes=4,
        use_attention=False,
    ).to(device)
    encoder.load_state_dict(torch.load(v0_path, map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("[LatentSemanticsAudit] Frozen PhotonV0 loaded.")

    # 2. Load Test dataset (75 sequences)
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
    _, _, test_loader = adapter.get_dataloaders(batch_size=75)
    test_batch = next(iter(test_loader))
    x_clean = test_batch["features"].to(device)  # [75, 16, 64]
    B, T, D = x_clean.shape

    with torch.no_grad():
        z_clean, _ = encoder.extract_latents(x_clean)  # [75, 16, 64]

    # 3. Apply Group Perturbations
    noise_std = float(torch.std(x_clean).item()) * 0.5

    # Range perturbation (indices 0:30)
    x_range = x_clean.clone()
    torch.manual_seed(42)
    x_range[:, :, 0:30] += torch.randn_like(x_range[:, :, 0:30]) * noise_std

    # Doppler perturbation (indices 30:60)
    x_doppler = x_clean.clone()
    torch.manual_seed(42)
    x_doppler[:, :, 30:60] += torch.randn_like(x_doppler[:, :, 30:60]) * noise_std

    # Summary perturbation (indices 60:64)
    x_summary = x_clean.clone()
    torch.manual_seed(42)
    x_summary[:, :, 60:64] += torch.randn_like(x_summary[:, :, 60:64]) * noise_std

    with torch.no_grad():
        z_range, _ = encoder.extract_latents(x_range)
        z_doppler, _ = encoder.extract_latents(x_doppler)
        z_summary, _ = encoder.extract_latents(x_summary)

    # 4. Measure Channel-wise Sensitivity
    # S_group(channel_k) = mean(|z_perturbed[:, :, k] - z_clean[:, :, k]|)
    sens_range = torch.mean(torch.abs(z_range - z_clean), dim=(0, 1)).cpu().numpy()  # [64]
    sens_doppler = torch.mean(torch.abs(z_doppler - z_clean), dim=(0, 1)).cpu().numpy()  # [64]
    sens_summary = torch.mean(torch.abs(z_summary - z_clean), dim=(0, 1)).cpu().numpy()  # [64]

    total_sens = sens_range + sens_doppler + sens_summary + 1e-8
    ratio_range = sens_range / total_sens
    ratio_doppler = sens_doppler / total_sens
    ratio_summary = sens_summary / total_sens

    # Check slice semantics
    # Slice 0:30
    mean_r_in_r_slice = float(np.mean(ratio_range[0:30]))
    mean_d_in_r_slice = float(np.mean(ratio_doppler[0:30]))
    mean_s_in_r_slice = float(np.mean(ratio_summary[0:30]))

    # Slice 30:60
    mean_r_in_d_slice = float(np.mean(ratio_range[30:60]))
    mean_d_in_d_slice = float(np.mean(ratio_doppler[30:60]))
    mean_s_in_d_slice = float(np.mean(ratio_summary[30:60]))

    # Overall ratios
    overall_mean_r = float(np.mean(ratio_range))
    overall_mean_d = float(np.mean(ratio_doppler))
    overall_mean_s = float(np.mean(ratio_summary))

    print("=============================================================")
    print("           LATENT CHANNEL SENSITIVITY AUDIT RESULTS          ")
    print("=============================================================")
    print(f"Slice [0:30]   (Assumed Range):   Range={mean_r_in_r_slice*100:.1f}%, Doppler={mean_d_in_r_slice*100:.1f}%, Summary={mean_s_in_r_slice*100:.1f}%")
    print(f"Slice [30:60]  (Assumed Doppler): Range={mean_r_in_d_slice*100:.1f}%, Doppler={mean_d_in_d_slice*100:.1f}%, Summary={mean_s_in_d_slice*100:.1f}%")
    print(f"Slice [60:64]  (Assumed Summary): Range={np.mean(ratio_range[60:64])*100:.1f}%, Doppler={np.mean(ratio_doppler[60:64])*100:.1f}%, Summary={np.mean(ratio_summary[60:64])*100:.1f}%")
    print(f"Overall Latent Sensitivity:      Range={overall_mean_r*100:.1f}%, Doppler={overall_mean_d*100:.1f}%, Summary={overall_mean_s*100:.1f}%")
    print("=============================================================")

    # Determine entanglement
    is_mixed = (mean_d_in_r_slice > 0.15) or (mean_r_in_d_slice > 0.15) or (abs(mean_r_in_r_slice - mean_r_in_d_slice) < 0.20)
    mapping_status = "MIXED" if is_mixed else "PHYSICALLY DISENTANGLED"

    print(f"PhotonV0 Latent Mapping: {mapping_status}")
    print(f"Direct z[0:30] Range Interpretation: {'INVALID' if is_mixed else 'VALID'}")
    print(f"Direct z[30:60] Doppler Interpretation: {'INVALID' if is_mixed else 'VALID'}")

    # 5. Visualizations
    results_dir = REPO_ROOT / "results" / "photon_v2"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: Absolute Sensitivity per Channel
    channels = np.arange(64)
    width = 0.28
    axes[0].bar(channels - width, sens_range, width=width, label="Range Input Perturbation", color="#1f77b4")
    axes[0].bar(channels, sens_doppler, width=width, label="Doppler Input Perturbation", color="#ff7f0e")
    axes[0].bar(channels + width, sens_summary, width=width, label="Summary Input Perturbation", color="#2ca02c")
    axes[0].axvline(29.5, color="black", linestyle="--", alpha=0.6, label="Slice Boundaries (30, 60)")
    axes[0].axvline(59.5, color="black", linestyle="--", alpha=0.6)
    axes[0].set_title("PhotonV0 Latent Channel Sensitivity to Input Feature Groups", fontweight="bold")
    axes[0].set_xlabel("Latent Channel Index (0 to 63)")
    axes[0].set_ylabel("Mean Absolute Latent Shift ||Δz||")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Bottom: Relative Sensitivity Ratio per Channel (Stacked)
    axes[1].bar(channels, ratio_range, label="Range Ratio", color="#1f77b4", alpha=0.85)
    axes[1].bar(channels, ratio_doppler, bottom=ratio_range, label="Doppler Ratio", color="#ff7f0e", alpha=0.85)
    axes[1].bar(channels, ratio_summary, bottom=ratio_range + ratio_doppler, label="Summary Ratio", color="#2ca02c", alpha=0.85)
    axes[1].axvline(29.5, color="white", linestyle="--", lw=1.5)
    axes[1].axvline(59.5, color="white", linestyle="--", lw=1.5)
    axes[1].set_title("Relative Composition of Learned Latent Channels (Demonstrating Channel Mixing)", fontweight="bold")
    axes[1].set_xlabel("Latent Channel Index (0 to 63)")
    axes[1].set_ylabel("Sensitivity Ratio (0 to 1.0)")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(results_dir / "latent_sensitivity.png", dpi=200)
    plt.close()

    # 6. Save JSON and Markdown Audit
    audit_data = {
        "mapping_status": mapping_status,
        "is_mixed": is_mixed,
        "slice_0_30": {
            "range_ratio": mean_r_in_r_slice,
            "doppler_ratio": mean_d_in_r_slice,
            "summary_ratio": mean_s_in_r_slice,
        },
        "slice_30_60": {
            "range_ratio": mean_r_in_d_slice,
            "doppler_ratio": mean_d_in_d_slice,
            "summary_ratio": mean_s_in_d_slice,
        },
        "slice_60_64": {
            "range_ratio": float(np.mean(ratio_range[60:64])),
            "doppler_ratio": float(np.mean(ratio_doppler[60:64])),
            "summary_ratio": float(np.mean(ratio_summary[60:64])),
        },
        "overall": {
            "range_ratio": overall_mean_r,
            "doppler_ratio": overall_mean_d,
            "summary_ratio": overall_mean_s,
        },
    }

    with open(results_dir / "latent_semantics_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    return audit_data


if __name__ == "__main__":
    run_semantics_audit()
