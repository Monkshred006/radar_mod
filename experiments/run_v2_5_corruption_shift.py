"""PhotonShield AI -- Phase V2.5 Variable-Corruption Robustness Experiment.

Tests whether training with variable temporal dropout (p in {0.10, 0.20, 0.30, 0.40, 0.50})
improves robustness at high dropout levels compared to fixed p=0.20 training (V2.3).

Key changes from V2.3/V2.4:
1. Variable corruption: each training batch samples p uniformly from {0.10, 0.20, 0.30, 0.40, 0.50}
2. Checkpoint selection: Validation Macro-F1 (fixes MSE/F1 mismatch found in audit)
3. No gap-aware weighting (gap_alpha=0, validated V2.3 physics loss)
4. Gap distribution telemetry logged per epoch

Tiny experiment: 10 training sequences, seed 42, single run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter
from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_05_latent_diffusion.denoiser import LightweightDenoiser
from module_05_latent_diffusion.scheduler import DDPMScheduler
from module_05_latent_diffusion.corruption import RadarLatentCorruption
from module_05_latent_diffusion.losses import DiffusionLoss
from module_06_physics.radar_constants import DT
from module_06_physics.latent_physics_head import LatentPhysicsHead
from module_06_physics.physics_losses import RadarPhysicsLoss

CLASS_NAMES = ["Empty", "Pedestrian", "Cyclist", "Vehicle"]
DROPOUT_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
EVAL_DROPOUT_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_gap_statistics(mask: torch.Tensor) -> Dict[str, float]:
    """Compute gap-length distribution statistics from a corruption mask [B, T, 1]."""
    obs = mask[:, :, 0].cpu().numpy()  # [B, T]
    all_gaps = []
    for b in range(obs.shape[0]):
        gap_len = 0
        for t in range(obs.shape[1]):
            if obs[b, t] == 0:
                gap_len += 1
            else:
                if gap_len > 0:
                    all_gaps.append(gap_len)
                gap_len = 0
        if gap_len > 0:
            all_gaps.append(gap_len)

    if len(all_gaps) == 0:
        return {
            "num_missing_frames": 0, "num_gaps": 0,
            "mean_gap_length": 0.0, "median_gap_length": 0.0, "max_gap_length": 0,
            "pct_gaps_ge3": 0.0, "pct_gaps_ge4": 0.0, "pct_gaps_ge5": 0.0,
        }

    total_missing = int(np.sum(obs == 0))
    arr = np.array(all_gaps)
    return {
        "num_missing_frames": total_missing,
        "num_gaps": len(all_gaps),
        "mean_gap_length": float(np.mean(arr)),
        "median_gap_length": float(np.median(arr)),
        "max_gap_length": int(np.max(arr)),
        "pct_gaps_ge3": float(100.0 * np.sum(arr >= 3) / len(arr)),
        "pct_gaps_ge4": float(100.0 * np.sum(arr >= 4) / len(arr)),
        "pct_gaps_ge5": float(100.0 * np.sum(arr >= 5) / len(arr)),
    }


def evaluate_model(
    denoiser: nn.Module,
    physics_head: nn.Module,
    scheduler: DDPMScheduler,
    encoder: PhotonV0,
    physics_loss_module: RadarPhysicsLoss,
    data_loader: DataLoader,
    corr_op: RadarLatentCorruption,
    device: torch.device,
) -> Dict[str, float]:
    """Run deterministic evaluation over a dataset split."""
    denoiser.eval()
    physics_head.eval()
    encoder.eval()

    sum_miss_mse = 0.0
    sum_full_mse = 0.0
    sum_r_mae = 0.0
    sum_v_mae = 0.0
    sum_kin_res = 0.0
    sum_acc_res = 0.0
    sum_phys_loss = 0.0
    total_samples = 0

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            x_clean = batch["features"].to(device)
            y_cls = batch["classification"].to(device)
            B = x_clean.shape[0]

            z0_clean, _ = encoder.extract_latents(x_clean)
            zc, mask = corr_op(z0_clean)

            z_hat = scheduler.reconstruct(
                denoiser=denoiser, condition=zc, mask=mask,
                num_inference_steps=50, deterministic=True,
            )

            diff_sq = (z_hat - z0_clean) ** 2
            full_mse = torch.mean(diff_sq)
            missing_mask = (1.0 - mask)
            missing_count = torch.sum(missing_mask)
            miss_mse = torch.sum(diff_sq * missing_mask) / (missing_count * z0_clean.shape[-1]) if missing_count > 0 else torch.tensor(0.0, device=device)

            obs_pred = physics_head(z_hat)
            r_hat = obs_pred["range"]
            v_hat = obs_pred["velocity"]

            r_gt = physics_loss_module.raw_extractor.extract_range(x_clean[..., 0:30])
            v_gt = physics_loss_module.raw_extractor.extract_velocity(x_clean[..., 30:60])
            r_mae = torch.mean(torch.abs(r_hat - r_gt))
            v_mae = torch.mean(torch.abs(v_hat - v_gt))

            p_loss, p_comp = physics_loss_module(z_hat, x_clean=None)
            kin_res = torch.mean(torch.abs(p_comp["kin_residual"]))
            acc_res = torch.mean(torch.abs(p_comp["acceleration"]))

            pooled = z_hat[:, -1, :]
            logits = encoder.classification_head(pooled)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            sum_miss_mse += miss_mse.item() * B
            sum_full_mse += full_mse.item() * B
            sum_r_mae += r_mae.item() * B
            sum_v_mae += v_mae.item() * B
            sum_kin_res += kin_res.item() * B
            sum_acc_res += acc_res.item() * B
            sum_phys_loss += p_loss.item() * B
            total_samples += B

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(y_cls.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    try:
        auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        auroc = 0.5

    n = max(total_samples, 1)
    result = {
        "missing_mse": sum_miss_mse / n,
        "full_mse": sum_full_mse / n,
        "range_mae": sum_r_mae / n,
        "velocity_mae": sum_v_mae / n,
        "kinematic_residual": sum_kin_res / n,
        "acceleration_residual": sum_acc_res / n,
        "physics_loss": sum_phys_loss / n,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "auroc": auroc,
    }
    for i, c_name in enumerate(CLASS_NAMES):
        result[f"f1_{c_name.lower()}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0
    return result


def run_v2_5_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V2.5] Device: {device}")

    results_dir = REPO_ROOT / "results" / "photon_v2"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen PhotonV0
    v0_path = REPO_ROOT / "checkpoints" / "v0_frozen" / "best_model.pt"
    encoder = PhotonV0(
        input_dim=64, hidden_dim=64, num_layers=2,
        sequence_length=16, num_classes=4, use_attention=False,
    ).to(device)
    encoder.load_state_dict(torch.load(v0_path, map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("[V2.5] Frozen PhotonV0 loaded.")

    # 2. Data
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
    print(f"[V2.5] Dataset: 10 Train, {len(val_loader.dataset)} Val.")

    # V1 checkpoint
    v1_ckpt_path = REPO_ROOT / "results" / "photon_v1" / "full_training" / "best_model.pt"
    if not v1_ckpt_path.exists():
        v1_ckpt_path = REPO_ROOT / "checkpoints" / "v1_diffusion" / "best_diffusion.pt"

    scheduler = DDPMScheduler(num_train_timesteps=50, beta_schedule="linear").to(device)

    # ================================================================
    # PART A: Load V2.3 Control (fixed 20% dropout, retrain with F1 ckpt)
    # ================================================================
    print("\n" + "=" * 60)
    print(" TRAINING V2.3 CONTROL (fixed p=0.20, F1-based checkpoint)")
    print("=" * 60)

    set_seed(42)
    v23_denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2).to(device)
    v23_denoiser.load_state_dict(torch.load(v1_ckpt_path, map_location=device))
    v23_physics_head = LatentPhysicsHead(latent_dim=64, hidden_dim=32).to(device)
    v23_physics_loss = RadarPhysicsLoss(
        dt=DT, velocity_sign=1, lambda_kin=1.0, lambda_acc=0.1,
        lambda_energy=0.1, lambda_align=0.5, gap_alpha=0.0,
        physics_head=v23_physics_head,
    ).to(device)
    diff_loss_fn = DiffusionLoss(lambda_diff=1.0, lambda_recon=0.5, lambda_missing=1.0)
    corr_fixed = RadarLatentCorruption({"enabled": True, "frame_dropout": {"enabled": True, "probability": 0.20}})

    v23_params = list(v23_denoiser.parameters()) + list(v23_physics_head.parameters())
    v23_optimizer = AdamW(v23_params, lr=5e-4, weight_decay=1e-4)
    v23_lr_sched = CosineAnnealingLR(v23_optimizer, T_max=50, eta_min=1e-5)
    v23_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    v23_best_f1 = -1.0
    v23_best_epoch = 0
    v23_patience = 0
    v23_ckpt_dir = REPO_ROOT / "checkpoints" / "v2_physics" / "v25_experiment" / "v23_control"
    v23_ckpt_dir.mkdir(parents=True, exist_ok=True)
    v23_ckpt_path = v23_ckpt_dir / "best_model.pt"

    for epoch in range(1, 51):
        v23_denoiser.train()
        v23_physics_head.train()
        for batch in train_subset_loader:
            x_clean = batch["features"].to(device)
            B = x_clean.shape[0]
            v23_optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    z0, _ = encoder.extract_latents(x_clean)
                    zc, mask = corr_fixed(z0)
                t_step = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
                noise = torch.randn_like(z0)
                z_t = scheduler.add_noise(z0, noise, t_step)
                noise_pred = v23_denoiser(z_t, zc, t_step, mask=mask)
                z0_pred = scheduler.predict_z0_from_eps(z_t, noise_pred, t_step)
                sqrt_a = scheduler.sqrt_alphas_cumprod[t_step].view(-1, 1, 1)
                l_v1, _ = diff_loss_fn(noise_pred, noise, z0_pred, z0, mask, sqrt_a)
                l_phys, _ = v23_physics_loss(z0_pred, x_clean=x_clean, mask=mask)
                total_loss = l_v1 + 0.01 * l_phys
            if device.type == "cuda":
                v23_scaler.scale(total_loss).backward()
                v23_scaler.unscale_(v23_optimizer)
                torch.nn.utils.clip_grad_norm_(v23_params, max_norm=1.0)
                v23_scaler.step(v23_optimizer)
                v23_scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(v23_params, max_norm=1.0)
                v23_optimizer.step()
        v23_lr_sched.step()

        # Validation (F1-based checkpoint selection)
        val_res = evaluate_model(v23_denoiser, v23_physics_head, scheduler, encoder,
                                 v23_physics_loss, val_loader, corr_fixed, device)
        if val_res["macro_f1"] > v23_best_f1:
            v23_best_f1 = val_res["macro_f1"]
            v23_best_epoch = epoch
            v23_patience = 0
            torch.save({"denoiser": v23_denoiser.state_dict(),
                        "physics_head": v23_physics_head.state_dict()}, v23_ckpt_path)
        else:
            v23_patience += 1
            if v23_patience >= 10:
                print(f"  V2.3 early stop at epoch {epoch}")
                break
        if epoch % 5 == 0 or epoch == 1:
            print(f"  V2.3 Epoch {epoch:02d} | Val F1: {val_res['macro_f1']:.4f}, MSE: {val_res['missing_mse']:.4f}")

    print(f"  V2.3 Best epoch: {v23_best_epoch} (F1: {v23_best_f1:.4f})")
    v23_ckpt = torch.load(v23_ckpt_path, map_location=device)
    v23_denoiser.load_state_dict(v23_ckpt["denoiser"])
    v23_physics_head.load_state_dict(v23_ckpt["physics_head"])

    # ================================================================
    # PART B: Train V2.5 (variable corruption)
    # ================================================================
    print("\n" + "=" * 60)
    print(" TRAINING V2.5 (variable p in {10,20,30,40,50}%, F1 ckpt)")
    print("=" * 60)

    set_seed(42)
    v25_denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2).to(device)
    v25_denoiser.load_state_dict(torch.load(v1_ckpt_path, map_location=device))
    v25_physics_head = LatentPhysicsHead(latent_dim=64, hidden_dim=32).to(device)
    v25_physics_loss = RadarPhysicsLoss(
        dt=DT, velocity_sign=1, lambda_kin=1.0, lambda_acc=0.1,
        lambda_energy=0.1, lambda_align=0.5, gap_alpha=0.0,
        physics_head=v25_physics_head,
    ).to(device)

    v25_params = list(v25_denoiser.parameters()) + list(v25_physics_head.parameters())
    v25_optimizer = AdamW(v25_params, lr=5e-4, weight_decay=1e-4)
    v25_lr_sched = CosineAnnealingLR(v25_optimizer, T_max=50, eta_min=1e-5)
    v25_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    v25_best_f1 = -1.0
    v25_best_epoch = 0
    v25_patience = 0
    v25_ckpt_dir = REPO_ROOT / "checkpoints" / "v2_physics" / "v25_experiment" / "v25_variable"
    v25_ckpt_dir.mkdir(parents=True, exist_ok=True)
    v25_ckpt_path = v25_ckpt_dir / "best_model.pt"

    # Gap distribution telemetry
    gap_telemetry = {"per_epoch": [], "aggregate": {}}

    # Variable corruption operators (one per dropout level)
    corr_operators = {
        p: RadarLatentCorruption({"enabled": True, "frame_dropout": {"enabled": True, "probability": p}})
        for p in DROPOUT_LEVELS
    }

    # Use mid-range evaluation corruption for validation checkpoint selection
    corr_val = RadarLatentCorruption({"enabled": True, "frame_dropout": {"enabled": True, "probability": 0.30}})

    for epoch in range(1, 51):
        v25_denoiser.train()
        v25_physics_head.train()

        epoch_gap_stats = []
        epoch_p_used = []

        for batch in train_subset_loader:
            x_clean = batch["features"].to(device)
            B = x_clean.shape[0]

            # Sample random dropout probability for this batch
            p_train = random.choice(DROPOUT_LEVELS)
            epoch_p_used.append(p_train)
            corr_train = corr_operators[p_train]

            v25_optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    z0, _ = encoder.extract_latents(x_clean)
                    zc, mask = corr_train(z0)

                # Record gap statistics
                epoch_gap_stats.append(compute_gap_statistics(mask))

                t_step = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
                noise = torch.randn_like(z0)
                z_t = scheduler.add_noise(z0, noise, t_step)
                noise_pred = v25_denoiser(z_t, zc, t_step, mask=mask)
                z0_pred = scheduler.predict_z0_from_eps(z_t, noise_pred, t_step)
                sqrt_a = scheduler.sqrt_alphas_cumprod[t_step].view(-1, 1, 1)
                l_v1, _ = diff_loss_fn(noise_pred, noise, z0_pred, z0, mask, sqrt_a)
                l_phys, _ = v25_physics_loss(z0_pred, x_clean=x_clean, mask=mask)
                total_loss = l_v1 + 0.01 * l_phys

            if device.type == "cuda":
                v25_scaler.scale(total_loss).backward()
                v25_scaler.unscale_(v25_optimizer)
                torch.nn.utils.clip_grad_norm_(v25_params, max_norm=1.0)
                v25_scaler.step(v25_optimizer)
                v25_scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(v25_params, max_norm=1.0)
                v25_optimizer.step()

        v25_lr_sched.step()

        # Epoch gap telemetry
        if epoch_gap_stats:
            epoch_telem = {
                "epoch": epoch,
                "dropout_p_used": epoch_p_used,
                "mean_gap_length": float(np.mean([s["mean_gap_length"] for s in epoch_gap_stats])),
                "max_gap_length": int(max(s["max_gap_length"] for s in epoch_gap_stats)),
                "pct_gaps_ge3": float(np.mean([s["pct_gaps_ge3"] for s in epoch_gap_stats])),
                "pct_gaps_ge4": float(np.mean([s["pct_gaps_ge4"] for s in epoch_gap_stats])),
                "pct_gaps_ge5": float(np.mean([s["pct_gaps_ge5"] for s in epoch_gap_stats])),
            }
            gap_telemetry["per_epoch"].append(epoch_telem)

        # Validation with p=0.30 (mid-range, sees some gaps)
        val_res = evaluate_model(v25_denoiser, v25_physics_head, scheduler, encoder,
                                 v25_physics_loss, val_loader, corr_val, device)

        if val_res["macro_f1"] > v25_best_f1:
            v25_best_f1 = val_res["macro_f1"]
            v25_best_epoch = epoch
            v25_patience = 0
            torch.save({"denoiser": v25_denoiser.state_dict(),
                        "physics_head": v25_physics_head.state_dict()}, v25_ckpt_path)
        else:
            v25_patience += 1
            if v25_patience >= 10:
                print(f"  V2.5 early stop at epoch {epoch}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  V2.5 Epoch {epoch:02d} | p={epoch_p_used[-1]:.2f} | "
                f"Val F1: {val_res['macro_f1']:.4f}, MSE: {val_res['missing_mse']:.4f}, "
                f"MaxGap: {epoch_telem['max_gap_length']}, "
                f"Gaps>=3: {epoch_telem['pct_gaps_ge3']:.0f}%"
            )

    print(f"  V2.5 Best epoch: {v25_best_epoch} (F1: {v25_best_f1:.4f})")

    # Aggregate gap telemetry
    all_mean_gaps = [e["mean_gap_length"] for e in gap_telemetry["per_epoch"]]
    all_max_gaps = [e["max_gap_length"] for e in gap_telemetry["per_epoch"]]
    all_ge3 = [e["pct_gaps_ge3"] for e in gap_telemetry["per_epoch"]]
    all_ge5 = [e["pct_gaps_ge5"] for e in gap_telemetry["per_epoch"]]
    gap_telemetry["aggregate"] = {
        "overall_mean_gap": float(np.mean(all_mean_gaps)),
        "overall_max_gap": int(max(all_max_gaps)),
        "overall_pct_ge3": float(np.mean(all_ge3)),
        "overall_pct_ge5": float(np.mean(all_ge5)),
    }
    with open(results_dir / "v2_5_gap_distribution.json", "w") as f:
        json.dump(gap_telemetry, f, indent=2)
    print(f"\n[V2.5] Gap distribution saved: v2_5_gap_distribution.json")

    # Load best V2.5 checkpoint
    v25_ckpt = torch.load(v25_ckpt_path, map_location=device)
    v25_denoiser.load_state_dict(v25_ckpt["denoiser"])
    v25_physics_head.load_state_dict(v25_ckpt["physics_head"])

    # ================================================================
    # PART C: Evaluate V1, V2.3, V2.5 at all dropout levels
    # ================================================================
    print("\n" + "=" * 60)
    print(" MULTI-DROPOUT EVALUATION: V1 vs V2.3 vs V2.5")
    print("=" * 60)

    # V1 baseline
    v1_denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2).to(device)
    v1_denoiser.load_state_dict(torch.load(v1_ckpt_path, map_location=device))
    v1_denoiser.eval()
    v1_physics_head = LatentPhysicsHead(latent_dim=64, hidden_dim=32).to(device)
    v1_physics_head.eval()
    v1_physics_loss = RadarPhysicsLoss(
        dt=DT, velocity_sign=1, gap_alpha=0.0, physics_head=v1_physics_head,
    ).to(device)

    models = {
        "V1": (v1_denoiser, v1_physics_head, v1_physics_loss),
        "V2.3_Fixed20": (v23_denoiser, v23_physics_head, v23_physics_loss),
        "V2.5_VarCorrupt": (v25_denoiser, v25_physics_head, v25_physics_loss),
    }

    all_rows = []
    for p_val in EVAL_DROPOUT_LEVELS:
        for model_name, (den, ph, pl) in models.items():
            set_seed(42)  # Identical corruption masks
            corr_eval = RadarLatentCorruption({"enabled": True, "frame_dropout": {"enabled": True, "probability": p_val}})
            res = evaluate_model(den, ph, scheduler, encoder, pl, val_loader, corr_eval, device)

            row = {
                "model": model_name, "dropout_p": p_val,
                "macro_f1": res["macro_f1"], "accuracy": res["accuracy"], "auroc": res["auroc"],
                "missing_mse": res["missing_mse"], "full_mse": res["full_mse"],
                "range_mae": res["range_mae"], "velocity_mae": res["velocity_mae"],
                "kinematic_residual": res["kinematic_residual"],
                "acceleration_residual": res["acceleration_residual"],
                "physics_loss": res["physics_loss"],
                "f1_empty": res.get("f1_empty", 0.0), "f1_pedestrian": res.get("f1_pedestrian", 0.0),
                "f1_cyclist": res.get("f1_cyclist", 0.0), "f1_vehicle": res.get("f1_vehicle", 0.0),
            }
            all_rows.append(row)

            print(
                f"[{model_name:15s} | p={int(p_val*100):02d}%] "
                f"F1: {res['macro_f1']:.4f}, Cyc: {res.get('f1_cyclist', 0):.4f}, "
                f"MSE: {res['missing_mse']:.4f}, R: {res['range_mae']:.3f}m, "
                f"Kin: {res['kinematic_residual']:.3f}"
            )

    # Save CSV
    csv_path = results_dir / "v2_5_corruption_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"\n[V2.5] CSV saved: {csv_path}")

    # ================================================================
    # PART D: Generate Plots
    # ================================================================

    def get_metric(model_name, metric):
        return [r[metric] for r in all_rows if r["model"] == model_name]

    p_arr = np.array(EVAL_DROPOUT_LEVELS) * 100

    # Plot 1: F1 vs Dropout
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p_arr, get_metric("V1", "macro_f1"), "o--", color="#1f77b4", label="V1 (No Physics)")
    ax.plot(p_arr, get_metric("V2.3_Fixed20", "macro_f1"), "s--", color="#d62728", label="V2.3 (Fixed 20%)")
    ax.plot(p_arr, get_metric("V2.5_VarCorrupt", "macro_f1"), "^-", color="#2ca02c", lw=2, label="V2.5 (Variable)")
    ax.set_xlabel("Temporal Frame Dropout (%)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Downstream Macro-F1 vs. Dropout Rate", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(results_dir / "v2_5_f1_vs_dropout.png", dpi=200)
    plt.close()

    # Plot 2: Missing MSE vs Dropout
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p_arr, get_metric("V1", "missing_mse"), "o--", color="#1f77b4", label="V1")
    ax.plot(p_arr, get_metric("V2.3_Fixed20", "missing_mse"), "s--", color="#d62728", label="V2.3 (Fixed 20%)")
    ax.plot(p_arr, get_metric("V2.5_VarCorrupt", "missing_mse"), "^-", color="#2ca02c", lw=2, label="V2.5 (Variable)")
    ax.set_xlabel("Temporal Frame Dropout (%)")
    ax.set_ylabel("Missing-Frame MSE")
    ax.set_title("Latent Reconstruction MSE vs. Dropout Rate", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(results_dir / "v2_5_missing_mse_vs_dropout.png", dpi=200)
    plt.close()

    # Plot 3: Physics (Kinematic) vs Dropout
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p_arr, get_metric("V1", "kinematic_residual"), "o--", color="#1f77b4", label="V1")
    ax.plot(p_arr, get_metric("V2.3_Fixed20", "kinematic_residual"), "s--", color="#d62728", label="V2.3 (Fixed 20%)")
    ax.plot(p_arr, get_metric("V2.5_VarCorrupt", "kinematic_residual"), "^-", color="#2ca02c", lw=2, label="V2.5 (Variable)")
    ax.set_xlabel("Temporal Frame Dropout (%)")
    ax.set_ylabel("Kinematic Residual |dR/dt - v| (m/s)")
    ax.set_title("Kinematic Consistency vs. Dropout Rate", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(results_dir / "v2_5_physics_vs_dropout.png", dpi=200)
    plt.close()

    # Plot 4: Cyclist F1 vs Dropout
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p_arr, get_metric("V1", "f1_cyclist"), "o--", color="#1f77b4", label="V1")
    ax.plot(p_arr, get_metric("V2.3_Fixed20", "f1_cyclist"), "s--", color="#d62728", label="V2.3 (Fixed 20%)")
    ax.plot(p_arr, get_metric("V2.5_VarCorrupt", "f1_cyclist"), "^-", color="#2ca02c", lw=2, label="V2.5 (Variable)")
    ax.set_xlabel("Temporal Frame Dropout (%)")
    ax.set_ylabel("Cyclist F1")
    ax.set_title("Cyclist Classification F1 vs. Dropout Rate", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(results_dir / "v2_5_cyclist_f1_vs_dropout.png", dpi=200)
    plt.close()

    # Plot 5: Training gap distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs_arr = [e["epoch"] for e in gap_telemetry["per_epoch"]]
    mean_gaps = [e["mean_gap_length"] for e in gap_telemetry["per_epoch"]]
    max_gaps = [e["max_gap_length"] for e in gap_telemetry["per_epoch"]]
    ge3_pct = [e["pct_gaps_ge3"] for e in gap_telemetry["per_epoch"]]
    ax.plot(epochs_arr, mean_gaps, "o-", color="#2ca02c", label="Mean Gap Length")
    ax.plot(epochs_arr, max_gaps, "s--", color="#d62728", label="Max Gap Length")
    ax2 = ax.twinx()
    ax2.plot(epochs_arr, ge3_pct, "^:", color="#ff7f0e", label="% Gaps >= 3")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Gap Length (frames)")
    ax2.set_ylabel("% Gaps >= 3 frames")
    ax.set_title("V2.5 Training Gap Distribution", fontweight="bold")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(results_dir / "v2_5_training_gap_distribution.png", dpi=200)
    plt.close()

    # ================================================================
    # PART E: Generate Report
    # ================================================================
    report_path = results_dir / "V2_5_CORRUPTION_SHIFT_REPORT.md"

    def r(model, p, metric):
        return [x[metric] for x in all_rows if x["model"] == model and x["dropout_p"] == p][0]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PhotonShield V2.5 -- Corruption-Distribution Robustness Report\n\n")
        f.write("## Hypothesis\n\n")
        f.write("Training with variable corruption ($p \\in \\{0.10, 0.20, 0.30, 0.40, 0.50\\}$) ")
        f.write("improves robustness at high dropout levels because the model explicitly sees long ")
        f.write("missing-frame gaps during training.\n\n")

        f.write("## Training Configuration\n\n")
        f.write("| Setting | V2.3 Control | V2.5 Variable |\n")
        f.write("|---|---|---|\n")
        f.write("| Training dropout | Fixed $p=0.20$ | Uniform $p \\in \\{0.10,...,0.50\\}$ |\n")
        f.write(f"| Checkpoint selection | **Macro-F1** (fixed) | **Macro-F1** (fixed) |\n")
        f.write(f"| Best epoch | {v23_best_epoch} | {v25_best_epoch} |\n")
        f.write(f"| Best val F1 | {v23_best_f1:.4f} | {v25_best_f1:.4f} |\n")
        f.write(f"| $\\lambda_{{physics}}$ | 0.01 | 0.01 |\n")
        f.write(f"| gap_alpha | 0.0 (none) | 0.0 (none) |\n\n")

        f.write("## Gap Distribution Telemetry\n\n")
        agg = gap_telemetry["aggregate"]
        f.write(f"- Overall mean gap length: **{agg['overall_mean_gap']:.2f} frames**\n")
        f.write(f"- Maximum gap observed: **{agg['overall_max_gap']} frames**\n")
        f.write(f"- Percentage of gaps >= 3 frames: **{agg['overall_pct_ge3']:.1f}%**\n")
        f.write(f"- Percentage of gaps >= 5 frames: **{agg['overall_pct_ge5']:.1f}%**\n\n")

        f.write("## Multi-Dropout Comparison\n\n")
        f.write("| Dropout | Model | Macro-F1 | Cyclist F1 | Accuracy | Miss MSE | Range MAE | Kin Res |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for p_val in EVAL_DROPOUT_LEVELS:
            for mn in ["V1", "V2.3_Fixed20", "V2.5_VarCorrupt"]:
                bold = "**" if mn == "V2.5_VarCorrupt" else ""
                f.write(
                    f"| {int(p_val*100)}% | {bold}{mn}{bold} | "
                    f"`{r(mn,p_val,'macro_f1'):.4f}` | `{r(mn,p_val,'f1_cyclist'):.4f}` | "
                    f"`{r(mn,p_val,'accuracy')*100:.1f}%` | `{r(mn,p_val,'missing_mse'):.4f}` | "
                    f"`{r(mn,p_val,'range_mae'):.3f}m` | `{r(mn,p_val,'kinematic_residual'):.3f}` |\n"
                )
            f.write("| | | | | | | | |\n")

        # Cyclist analysis
        f.write("\n## Cyclist F1 Analysis\n\n")
        f.write("| Dropout | V2.3 Cyclist F1 | V2.5 Cyclist F1 | Delta |\n")
        f.write("| :---: | :---: | :---: | :---: |\n")
        for p_val in [0.20, 0.30, 0.40, 0.50]:
            c23 = r("V2.3_Fixed20", p_val, "f1_cyclist")
            c25 = r("V2.5_VarCorrupt", p_val, "f1_cyclist")
            f.write(f"| {int(p_val*100)}% | `{c23:.4f}` | `{c25:.4f}` | `{c25-c23:+.4f}` |\n")

        # Success criteria
        f.write("\n## Success Criteria\n\n")
        c1 = r("V2.5_VarCorrupt", 0.40, "macro_f1") > r("V2.3_Fixed20", 0.40, "macro_f1") or \
             r("V2.5_VarCorrupt", 0.50, "macro_f1") > r("V2.3_Fixed20", 0.50, "macro_f1")
        c2 = r("V2.5_VarCorrupt", 0.50, "f1_cyclist") > r("V2.3_Fixed20", 0.50, "f1_cyclist") or \
             r("V2.5_VarCorrupt", 0.40, "f1_cyclist") > r("V2.3_Fixed20", 0.40, "f1_cyclist")
        c3 = r("V2.5_VarCorrupt", 0.50, "kinematic_residual") < r("V1", 0.50, "kinematic_residual")
        c4_10 = r("V2.5_VarCorrupt", 0.10, "macro_f1") >= r("V2.3_Fixed20", 0.10, "macro_f1") * 0.95
        c4_20 = r("V2.5_VarCorrupt", 0.20, "macro_f1") >= r("V2.3_Fixed20", 0.20, "macro_f1") * 0.95

        f.write(f"1. 40% or 50% F1 improves over V2.3: **{'PASS' if c1 else 'FAIL'}**\n")
        f.write(f"2. Cyclist F1 improves at high dropout: **{'PASS' if c2 else 'FAIL'}**\n")
        f.write(f"3. Physics consistency better than V1: **{'PASS' if c3 else 'FAIL'}**\n")
        f.write(f"4. 10%/20% performance not collapsed: **{'PASS' if (c4_10 and c4_20) else 'FAIL'}**\n\n")

        if c1 and c2 and c3 and c4_10 and c4_20:
            status = "PROMISING"
        elif c1 and not (c4_10 and c4_20):
            status = "PARTIAL"
        else:
            status = "FAILED"

        f.write(f"## FINAL STATUS: **{status}**\n")

    print(f"\n[V2.5] Report saved: {report_path}")
    print(f"\n{'='*60}")
    print(f" V2.5 CORRUPTION-DISTRIBUTION EXPERIMENT COMPLETE")
    print(f" FINAL STATUS: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_v2_5_experiment()
