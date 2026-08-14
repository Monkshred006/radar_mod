"""Run 1 full-data batch verification test through PhotonV0 on RTX 5050 GPU."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter

def run_single_batch_test():
    with open("configs/photon_v0_full.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running full-data single batch verification on device: {device}")

    adapter = RaDICaLDatasetAdapter(
        data_path=cfg["dataset"]["data_path"],
        sequence_length=16,
        feature_dim=64,
        num_classes=4,
        normalization="db",
        synthetic_fallback=False,
    )
    train_loader, val_loader, test_loader = adapter.get_dataloaders(batch_size=32)

    batch = next(iter(train_loader))
    x = batch["features"].to(device)
    y_det = batch["detection"].to(device)
    y_cls = batch["classification"].to(device)
    y_ano = batch["anomaly"].to(device)

    print(f"1. Batch tensor shape: {x.shape} (Expected: [32, 16, 64])")
    assert x.shape == (32, 16, 64), f"Unexpected shape: {x.shape}"
    
    nans = bool(torch.isnan(x).any().item())
    infs = bool(torch.isinf(x).any().item())
    print(f"2. NaNs in input: {nans}, Infs in input: {infs}")
    assert not nans and not infs, "NaN or Inf detected in batch input"

    valid_det = bool(((y_det >= 0) & (y_det <= 1)).all().item())
    valid_cls = bool(((y_cls >= 0) & (y_cls <= 3)).all().item())
    print(f"3. Labels valid: det in [0,1]: {valid_det}, cls in [0,3]: {valid_cls}")
    assert valid_det and valid_cls, "Invalid label ranges"

    model = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16, num_classes=4).to(device)
    model.train()

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(x, return_latents=True)
    
    det_shape = outputs["detection"].shape
    cls_shape = outputs["classification"].shape
    lat_shape = outputs["latent"].shape
    print(f"4. Forward pass successful: det={det_shape}, cls={cls_shape}, latent={lat_shape}")

    bce = torch.nn.BCELoss()
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    with torch.amp.autocast(device_type="cuda", enabled=False):
        loss = bce(outputs["detection"].float(), y_det.float()) + ce(outputs["classification"].float(), y_cls) + 0.5 * mse(outputs["anomaly"].float(), y_ano.float())
    print(f"5. Loss computed: {loss.item():.4f}")

    loss.backward()
    print("6. Backward pass successful: gradients calculated without errors!")
    print("================================================================")
    print(" ALL 6 FULL-DATA BATCH VERIFICATION CHECKS PASSED!")
    print("================================================================")

if __name__ == "__main__":
    run_single_batch_test()
