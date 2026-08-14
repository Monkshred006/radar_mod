"""PhotonShield V2.5 — Part 1: Training Pipeline Audit.

Diagnoses why V2.3 and V2.4 tiny experiments selected epoch 1.
Inspects: checkpoint metric, gradient norms, parameter deltas, optimizer steps.
"""

from __future__ import annotations
from pathlib import Path
import random
import sys
import json
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score

from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter
from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_05_latent_diffusion.denoiser import LightweightDenoiser
from module_05_latent_diffusion.scheduler import DDPMScheduler
from module_05_latent_diffusion.corruption import RadarLatentCorruption
from module_05_latent_diffusion.losses import DiffusionLoss
from module_06_physics.radar_constants import DT
from module_06_physics.latent_physics_head import LatentPhysicsHead
from module_06_physics.physics_losses import RadarPhysicsLoss


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def param_norm(model: nn.Module) -> float:
    return float(sum(p.data.norm().item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5)


def param_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.flatten() for p in model.parameters() if p.requires_grad])


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total += p.grad.data.norm().item() ** 2
    return float(total ** 0.5)


def run_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AUDIT] Device: {device}")

    set_seed(42)

    # Load frozen V0
    v0_path = REPO_ROOT / "checkpoints" / "v0_frozen" / "best_model.pt"
    encoder = PhotonV0(
        input_dim=64, hidden_dim=64, num_layers=2,
        sequence_length=16, num_classes=4, use_attention=False,
    ).to(device)
    encoder.load_state_dict(torch.load(v0_path, map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    frozen_params = sum(p.numel() for p in encoder.parameters())
    print(f"[AUDIT] Frozen V0 parameters: {frozen_params}")

    # Data
    adapter = RaDICaLDatasetAdapter(
        data_path="C:/Users/worka/research/photonpinn/data/radical",
        splits_dir="C:/Users/worka/research/photonpinn/data/radical/splits",
        sequence_length=16, feature_dim=64, num_classes=4,
        normalization="db", seed=42, synthetic_fallback=False,
    )
    full_train_loader, val_loader, _ = adapter.get_dataloaders(batch_size=16)
    train_subset_loader = DataLoader(
        Subset(full_train_loader.dataset, list(range(10))),
        batch_size=10, shuffle=True,
    )

    # V1 checkpoint
    v1_ckpt_path = REPO_ROOT / "results" / "photon_v1" / "full_training" / "best_model.pt"
    if not v1_ckpt_path.exists():
        v1_ckpt_path = REPO_ROOT / "checkpoints" / "v1_diffusion" / "best_diffusion.pt"

    # Initialize models
    denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2).to(device)
    denoiser.load_state_dict(torch.load(v1_ckpt_path, map_location=device))
    physics_head = LatentPhysicsHead(latent_dim=64, hidden_dim=32).to(device)

    denoiser_trainable = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    physics_trainable = sum(p.numel() for p in physics_head.parameters() if p.requires_grad)
    print(f"[AUDIT] Denoiser trainable params: {denoiser_trainable}")
    print(f"[AUDIT] Physics head trainable params: {physics_trainable}")
    print(f"[AUDIT] Total trainable params: {denoiser_trainable + physics_trainable}")

    # Save initial parameter state
    init_denoiser_vec = param_vector(denoiser).clone()
    init_physics_vec = param_vector(physics_head).clone()
    init_denoiser_norm = param_norm(denoiser)
    init_physics_norm = param_norm(physics_head)

    # Training setup
    scheduler = DDPMScheduler(num_train_timesteps=50, beta_schedule="linear").to(device)
    diff_loss_fn = DiffusionLoss(lambda_diff=1.0, lambda_recon=0.5, lambda_missing=1.0)
    physics_loss_fn = RadarPhysicsLoss(
        dt=DT, velocity_sign=1, lambda_kin=1.0, lambda_acc=0.1,
        lambda_energy=0.1, lambda_align=0.5, gap_alpha=0.0,
        physics_head=physics_head,
    ).to(device)
    corr_op = RadarLatentCorruption({"enabled": True, "frame_dropout": {"enabled": True, "probability": 0.20}})

    params = list(denoiser.parameters()) + list(physics_head.parameters())
    optimizer = AdamW(params, lr=5e-4, weight_decay=1e-4)
    lr_scheduler_obj = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Track per-epoch metrics
    audit_log = []
    total_optimizer_steps = 0

    print("\n" + "=" * 80)
    print(" TRAINING PIPELINE AUDIT (15 epochs, no early stopping)")
    print("=" * 80)

    for epoch in range(1, 16):
        denoiser.train()
        physics_head.train()

        epoch_grad_norms = []
        epoch_loss_v1 = []
        epoch_loss_phys = []
        epoch_loss_total = []

        for batch in train_subset_loader:
            x_clean = batch["features"].to(device)
            B = x_clean.shape[0]
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    z0, _ = encoder.extract_latents(x_clean)
                    zc, mask = corr_op(z0)

                t_step = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
                noise = torch.randn_like(z0)
                z_t = scheduler.add_noise(z0, noise, t_step)
                noise_pred = denoiser(z_t, zc, t_step, mask=mask)
                z0_pred = scheduler.predict_z0_from_eps(z_t, noise_pred, t_step)

                sqrt_alphas = scheduler.sqrt_alphas_cumprod[t_step].view(-1, 1, 1)
                l_v1, _ = diff_loss_fn(noise_pred, noise, z0_pred, z0, mask, sqrt_alphas)
                l_phys, _ = physics_loss_fn(z0_pred, x_clean=x_clean, mask=mask)
                total_loss = l_v1 + 0.01 * l_phys

            if device.type == "cuda":
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            total_optimizer_steps += 1
            epoch_grad_norms.append(grad_norm(denoiser) + grad_norm(physics_head))
            epoch_loss_v1.append(l_v1.item())
            epoch_loss_phys.append(l_phys.item())
            epoch_loss_total.append(total_loss.item())

        lr_scheduler_obj.step()

        # Parameter delta from initialization
        cur_denoiser_vec = param_vector(denoiser)
        cur_physics_vec = param_vector(physics_head)
        denoiser_delta = (cur_denoiser_vec - init_denoiser_vec).norm().item()
        physics_delta = (cur_physics_vec - init_physics_vec).norm().item()
        cur_denoiser_norm = param_norm(denoiser)
        cur_physics_norm = param_norm(physics_head)

        # Quick validation: Missing MSE and Macro-F1
        denoiser.eval()
        physics_head.eval()
        val_miss_mse_list = []
        val_preds_all = []
        val_targets_all = []
        with torch.no_grad():
            for batch in val_loader:
                x_clean = batch["features"].to(device)
                y_cls = batch["classification"].to(device)
                B = x_clean.shape[0]
                z0_clean, _ = encoder.extract_latents(x_clean)
                zc, mask = corr_op(z0_clean)
                z_hat = scheduler.reconstruct(denoiser=denoiser, condition=zc, mask=mask,
                                              num_inference_steps=50, deterministic=True)
                diff_sq = (z_hat - z0_clean) ** 2
                missing_mask = (1.0 - mask)
                missing_count = torch.sum(missing_mask)
                if missing_count > 0:
                    miss_mse = torch.sum(diff_sq * missing_mask) / (missing_count * z0_clean.shape[-1])
                else:
                    miss_mse = torch.tensor(0.0, device=device)
                val_miss_mse_list.append(miss_mse.item() * B)

                pooled = z_hat[:, -1, :]
                logits = encoder.classification_head(pooled)
                preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
                val_preds_all.extend(preds.cpu().numpy().tolist())
                val_targets_all.extend(y_cls.cpu().numpy().tolist())

        val_miss_mse = sum(val_miss_mse_list) / len(val_loader.dataset)
        val_macro_f1 = float(f1_score(val_targets_all, val_preds_all, average="macro", zero_division=0))

        entry = {
            "epoch": epoch,
            "train_loss_total": float(np.mean(epoch_loss_total)),
            "train_loss_v1": float(np.mean(epoch_loss_v1)),
            "train_loss_phys": float(np.mean(epoch_loss_phys)),
            "mean_grad_norm": float(np.mean(epoch_grad_norms)),
            "denoiser_param_norm": cur_denoiser_norm,
            "physics_param_norm": cur_physics_norm,
            "denoiser_delta_from_init": denoiser_delta,
            "physics_delta_from_init": physics_delta,
            "val_missing_mse": val_miss_mse,
            "val_macro_f1": val_macro_f1,
            "optimizer_steps": total_optimizer_steps,
            "lr": optimizer.param_groups[0]["lr"],
        }
        audit_log.append(entry)

        marker_mse = " <-- MSE best" if epoch == 1 or val_miss_mse < min(e["val_missing_mse"] for e in audit_log[:-1]) else ""
        marker_f1 = " <-- F1 best" if val_macro_f1 >= max(e["val_macro_f1"] for e in audit_log) else ""

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {entry['train_loss_total']:.4f} (V1: {entry['train_loss_v1']:.4f}, Phys: {entry['train_loss_phys']:.4f}) | "
            f"GradNorm: {entry['mean_grad_norm']:.4f} | "
            f"Delta_den: {denoiser_delta:.4f}, Delta_phys: {physics_delta:.4f} | "
            f"Val MSE: {val_miss_mse:.5f}{marker_mse} | "
            f"Val F1: {val_macro_f1:.4f}{marker_f1}"
        )

    # Summary
    print("\n" + "=" * 80)
    print(" AUDIT SUMMARY")
    print("=" * 80)
    print(f"Frozen parameters (V0):     {frozen_params}")
    print(f"Trainable denoiser params:  {denoiser_trainable}")
    print(f"Trainable physics params:   {physics_trainable}")
    print(f"Total trainable params:     {denoiser_trainable + physics_trainable}")
    print(f"Total optimizer steps:      {total_optimizer_steps}")
    print(f"Initial denoiser norm:      {init_denoiser_norm:.4f}")
    print(f"Final denoiser norm:        {param_norm(denoiser):.4f}")
    print(f"Initial physics norm:       {init_physics_norm:.4f}")
    print(f"Final physics norm:         {param_norm(physics_head):.4f}")
    print(f"Denoiser delta (15 epochs): {(param_vector(denoiser) - init_denoiser_vec).norm().item():.4f}")
    print(f"Physics delta (15 epochs):  {(param_vector(physics_head) - init_physics_vec).norm().item():.4f}")

    # Identify best epoch by each metric
    best_mse_epoch = min(audit_log, key=lambda x: x["val_missing_mse"])
    best_f1_epoch = max(audit_log, key=lambda x: x["val_macro_f1"])
    print(f"\nBest MSE epoch:  {best_mse_epoch['epoch']} (MSE = {best_mse_epoch['val_missing_mse']:.5f})")
    print(f"Best F1 epoch:   {best_f1_epoch['epoch']} (F1 = {best_f1_epoch['val_macro_f1']:.4f})")

    mismatch = best_mse_epoch["epoch"] != best_f1_epoch["epoch"]
    if mismatch:
        print(f"\n>>> CHECKPOINT SELECTION MISMATCH DETECTED <<<")
        print(f"MSE-optimal epoch {best_mse_epoch['epoch']}: F1 = {best_mse_epoch['val_macro_f1']:.4f}")
        print(f"F1-optimal epoch {best_f1_epoch['epoch']}: MSE = {best_f1_epoch['val_missing_mse']:.5f}")
        print(f"Delta F1 = {best_f1_epoch['val_macro_f1'] - best_mse_epoch['val_macro_f1']:+.4f}")
        print(f"RECOMMENDATION: Use Macro-F1 checkpoint selection for V2.5")
    else:
        print(f"\nNo mismatch: both metrics agree on epoch {best_mse_epoch['epoch']}")

    # Verify parameters actually changed
    final_delta = (param_vector(denoiser) - init_denoiser_vec).norm().item()
    if final_delta < 1e-6:
        print("\n>>> BUG: Denoiser parameters did NOT change from initialization! <<<")
    else:
        print(f"\nCONFIRMED: theta_epoch_15 != theta_epoch_0 (delta = {final_delta:.4f})")

    # Save audit JSON
    results_dir = REPO_ROOT / "results" / "photon_v2"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "v2_5_training_audit.json", "w") as f:
        json.dump({"audit_log": audit_log, "summary": {
            "frozen_params": frozen_params,
            "trainable_denoiser": denoiser_trainable,
            "trainable_physics": physics_trainable,
            "best_mse_epoch": best_mse_epoch["epoch"],
            "best_f1_epoch": best_f1_epoch["epoch"],
            "mismatch": mismatch,
            "final_denoiser_delta": final_delta,
        }}, f, indent=2)

    print(f"\n[AUDIT] Saved to: {results_dir / 'v2_5_training_audit.json'}")


if __name__ == "__main__":
    run_audit()
