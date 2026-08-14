"""Audit latent statistics of frozen PhotonV0 encoder on RaDICaL training set."""

from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter
from module_04_mamba_hybrid.photon_v0 import PhotonV0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Auditing latent statistics on {device}...")

    # Load frozen PhotonV0
    v0_path = REPO_ROOT / "checkpoints" / "v0_frozen" / "best_model.pt"
    encoder = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16, num_classes=4, use_attention=False).to(device)
    encoder.load_state_dict(torch.load(v0_path, map_location=device))
    encoder.eval()

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
    with torch.no_grad():
        for batch in train_loader:
            x = batch["features"].to(device)
            z0, _ = encoder.extract_latents(x)
            all_z0.append(z0.cpu())

    z0 = torch.cat(all_z0, dim=0) # [N, T, D]
    mean_val = float(z0.mean().item())
    std_val = float(z0.std().item())
    min_val = float(z0.min().item())
    max_val = float(z0.max().item())
    abs_mean = float(z0.abs().mean().item())

    print("================ LATENT AUDIT REPORT ================")
    print(f"Total training samples: {z0.shape[0]} sequences (shape: {z0.shape})")
    print(f"mean(z0)     : {mean_val:.6f}")
    print(f"std(z0)      : {std_val:.6f}")
    print(f"min(z0)      : {min_val:.6f}")
    print(f"max(z0)      : {max_val:.6f}")
    print(f"mean(|z0|)   : {abs_mean:.6f}")
    print("======================================================")

if __name__ == "__main__":
    main()
