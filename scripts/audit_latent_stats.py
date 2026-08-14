"""Audit latent statistics of frozen PhotonV0 encoder on RaDICaL training set."""

from __future__ import annotations
import sys
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter
from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_05_latent_diffusion.corruption import RadarLatentCorruption


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Auditing latent statistics on {device}...")

    # Load frozen PhotonV0
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

    corruption = RadarLatentCorruption({
        "enabled": True,
        "frame_dropout": {"enabled": True, "probability": 0.20},
    })

    # Load training data
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
    train_loader, _, _ = adapter.get_dataloaders(batch_size=32)

    all_z0 = []
    all_zc = []
    all_masks = []

    with torch.no_grad():
        for batch in train_loader:
            x = batch["features"].to(device)
            z0, _ = encoder.extract_latents(x)
            zc, mask = corruption(z0)

            all_z0.append(z0.cpu())
            all_zc.append(zc.cpu())
            all_masks.append(mask.cpu())

    z0 = torch.cat(all_z0, dim=0)   # [N, T, D]
    zc = torch.cat(all_zc, dim=0)   # [N, T, D]
    masks = torch.cat(all_masks, dim=0) # [N, T, 1]

    # Clean latent statistics
    z0_mean = float(z0.mean().item())
    z0_std = float(z0.std().item())
    z0_min = float(z0.min().item())
    z0_max = float(z0.max().item())
    z0_abs_mean = float(z0.abs().mean().item())

    # Corrupted latent statistics
    zc_mean = float(zc.mean().item())
    zc_std = float(zc.std().item())
    zc_min = float(zc.min().item())
    zc_max = float(zc.max().item())
    zc_abs_mean = float(zc.abs().mean().item())

    # Missing frame stats
    missing_ratio = float((1.0 - masks).mean().item())

    report_data = {
        "num_sequences": int(z0.shape[0]),
        "sequence_length": int(z0.shape[1]),
        "feature_dim": int(z0.shape[2]),
        "clean_latent_z0": {
            "mean": z0_mean,
            "std": z0_std,
            "min": z0_min,
            "max": z0_max,
            "mean_abs": z0_abs_mean,
        },
        "corrupted_latent_zc": {
            "mean": zc_mean,
            "std": zc_std,
            "min": zc_min,
            "max": zc_max,
            "mean_abs": zc_abs_mean,
        },
        "missing_frame_ratio": missing_ratio,
    }

    # Save outputs
    audit_dir = REPO_ROOT / "results" / "photon_v1" / "latent_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    json_path = audit_dir / "latent_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    md_path = audit_dir / "LATENT_AUDIT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PhotonShield AI — Latent Statistics Audit Report\n\n")
        f.write(f"- **Sequences**: {z0.shape[0]}\n")
        f.write(f"- **Shape**: `[B={z0.shape[0]}, T={z0.shape[1]}, D={z0.shape[2]}]`\n\n")
        f.write("### Clean Latent Statistics (z0)\n")
        f.write(f"- **Mean**: `{z0_mean:.6f}`\n")
        f.write(f"- **Std**: `{z0_std:.6f}`\n")
        f.write(f"- **Min**: `{z0_min:.6f}`\n")
        f.write(f"- **Max**: `{z0_max:.6f}`\n")
        f.write(f"- **Mean Absolute**: `{z0_abs_mean:.6f}`\n\n")
        f.write("### Corrupted Latent Statistics (zc)\n")
        f.write(f"- **Mean**: `{zc_mean:.6f}`\n")
        f.write(f"- **Std**: `{zc_std:.6f}`\n")
        f.write(f"- **Min**: `{zc_min:.6f}`\n")
        f.write(f"- **Max**: `{zc_max:.6f}`\n")
        f.write(f"- **Mean Absolute**: `{zc_abs_mean:.6f}`\n")
        f.write(f"- **Actual Frame Dropout Ratio**: `{missing_ratio:.4f}`\n")

    print("================ LATENT AUDIT REPORT ================")
    print(f"Total training samples: {z0.shape[0]} sequences (shape: {z0.shape})")
    print(f"Clean z0:   mean={z0_mean:.6f}, std={z0_std:.6f}, min={z0_min:.6f}, max={z0_max:.6f}, mean(|z0|)={z0_abs_mean:.6f}")
    print(f"Corrupt zc: mean={zc_mean:.6f}, std={zc_std:.6f}, min={zc_min:.6f}, max={zc_max:.6f}, mean(|zc|)={zc_abs_mean:.6f}")
    print(f"Saved audit reports to: {audit_dir}")
    print("======================================================")


if __name__ == "__main__":
    main()
