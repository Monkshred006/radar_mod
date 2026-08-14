"""PhotonShield AI -- Phase V2.4 Gap-Aware Physics Tiny Experiment.

Compares three models on 10 training sequences at 20%, 40%, 50% dropout:
- V1: Frozen V1 baseline (no physics)
- V2.3: Fixed physics (lambda=0.01, uniform gap weight)
- V2.4: Gap-aware physics (lambda=0.01, alpha=0.5)

Evaluates:
- Macro-F1, per-class F1 (especially Cyclist)
- Missing MSE
- Range MAE, Velocity MAE
- Kinematic residual

Checkpoint selection: Validation Missing-Frame MSE (matching tiny ablation convention).
"""

from __future__ import annotations

import csv
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
DROPOUT_LEVELS = [0.20, 0.40, 0.50]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
                denoiser=denoiser,
                condition=zc,
                mask=mask,
                num_inference_steps=50,
                deterministic=True,
            )

            # Reconstruction metrics
            diff_sq = (z_hat - z0_clean) ** 2
            full_mse = torch.mean(diff_sq)
            missing_mask = (1.0 - mask)
            missing_count = torch.sum(missing_mask)
            if missing_count > 0:
                miss_mse = torch.sum(diff_sq * missing_mask) / (missing_count * z0_clean.shape[-1])
            else:
                miss_mse = torch.tensor(0.0, device=device)

            # Physics metrics
            obs_pred = physics_head(z_hat)
            r_hat = obs_pred["range"]
            v_hat = obs_pred["velocity"]

            r_gt = physics_loss_module.raw_extractor.extract_range(x_clean[..., 0:30])
            v_gt = physics_loss_module.raw_extractor.extract_velocity(x_clean[..., 30:60])
            r_mae = torch.mean(torch.abs(r_hat - r_gt))
            v_mae = torch.mean(torch.abs(v_hat - v_gt))

            p_loss, p_comp = physics_loss_module(z_hat, x_clean=None)
            kin_res = torch.mean(torch.abs(p_comp["kin_residual"]))

            # Downstream perception
            pooled_latent = z_hat[:, -1, :]
            logits = encoder.classification_head(pooled_latent)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            sum_miss_mse += miss_mse.item() * B
            sum_full_mse += full_mse.item() * B
            sum_r_mae += r_mae.item() * B
            sum_v_mae += v_mae.item() * B
            sum_kin_res += kin_res.item() * B
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
        "physics_loss": sum_phys_loss / n,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "auroc": auroc,
    }
    for i, c_name in enumerate(CLASS_NAMES):
        result[f"f1_{c_name.lower()}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0
    return result


def train_v2_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    encoder: PhotonV0,
    v1_ckpt_path: Path,
    device: torch.device,
    save_dir: Path,
    gap_alpha: float,
    lambda_phys: float = 0.01,
    epochs: int = 50,
    patience: int = 10,
    model_label: str = "V2",
) -> Dict[str, Any]:
    """Train a V2 physics model (fixed or gap-aware) on 10 sequences."""
    print(f"\n========================================================")
    print(f" TRAINING {model_label}: lambda={lambda_phys:.2f}, gap_alpha={gap_alpha:.2f}")
    print(f"========================================================")

    set_seed(42)

    # Initialize from frozen V1
    denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2).to(device)
    denoiser.load_state_dict(torch.load(v1_ckpt_path, map_location=device))

    physics_head = LatentPhysicsHead(latent_dim=64, hidden_dim=32).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=50, beta_schedule="linear").to(device)
    diff_loss_fn = DiffusionLoss(lambda_diff=1.0, lambda_recon=0.5, lambda_missing=1.0)
    physics_loss_fn = RadarPhysicsLoss(
        dt=DT,
        velocity_sign=1,
        lambda_kin=1.0,
        lambda_acc=0.1,
        lambda_energy=0.1,
        lambda_align=0.5,
        gap_alpha=gap_alpha,
        physics_head=physics_head,
    ).to(device)
    corr_op = RadarLatentCorruption({"enabled": True, "frame_dropout": {"enabled": True, "probability": 0.20}})

    params = list(denoiser.parameters()) + list(physics_head.parameters())
    optimizer = AdamW(params, lr=5e-4, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_miss_mse = float("inf")
    best_metrics = {}
    best_epoch = 0
    patience_counter = 0

    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "best_model.pt"

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        denoiser.train()
        physics_head.train()

        sum_train_loss = 0.0
        sum_p_loss = 0.0
        n_train = 0

        for batch in train_loader:
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
                total_loss = l_v1 + lambda_phys * l_phys

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

            sum_train_loss += total_loss.item() * B
            sum_p_loss += l_phys.item() * B
            n_train += B

        lr_scheduler.step()

        # Validation
        val_res = evaluate_model(
            denoiser=denoiser,
            physics_head=physics_head,
            scheduler=scheduler,
            encoder=encoder,
            physics_loss_module=physics_loss_fn,
            data_loader=val_loader,
            corr_op=corr_op,
            device=device,
        )

        mean_train = sum_train_loss / max(n_train, 1)
        mean_p = sum_p_loss / max(n_train, 1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] Train: {mean_train:.4f} (Phys: {mean_p:.4f}) | "
            f"Val Miss MSE: {val_res['missing_mse']:.4f}, Macro-F1: {val_res['macro_f1']:.4f}, "
            f"R_MAE: {val_res['range_mae']:.2f}m, Kin: {val_res['kinematic_residual']:.2f}"
        )

        # Checkpoint selection: Validation Missing-Frame MSE
        if val_res["missing_mse"] < best_val_miss_mse:
            best_val_miss_mse = val_res["missing_mse"]
            best_metrics = val_res.copy()
            best_metrics["best_epoch"] = epoch
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "denoiser": denoiser.state_dict(),
                "physics_head": physics_head.state_dict(),
                "gap_alpha": gap_alpha,
                "lambda_physics": lambda_phys,
                "epoch": epoch,
                "metrics": val_res,
            }, best_ckpt_path)
            print(f"  --> Saved best checkpoint at epoch {epoch} (Miss MSE: {best_val_miss_mse:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} ({patience} epochs without improvement).")
                break

    best_metrics["training_time_s"] = round(time.perf_counter() - t_start, 2)
    best_metrics["model"] = model_label
    best_metrics["gap_alpha"] = gap_alpha
    best_metrics["best_epoch"] = best_epoch

    # Reload best checkpoint
    ckpt = torch.load(best_ckpt_path, map_location=device)
    denoiser.load_state_dict(ckpt["denoiser"])
    physics_head.load_state_dict(ckpt["physics_head"])

    return {
        "denoiser": denoiser,
        "physics_head": physics_head,
        "physics_loss_fn": physics_loss_fn,
        "metrics": best_metrics,
    }


def run_v2_4_gap_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V2.4 Gap-Aware Experiment] Device: {device}")

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
    print("[V2.4] Frozen PhotonV0 loaded.")

    # 2. Data loaders
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
    print(f"[V2.4] Dataset: 10 Train, {len(val_loader.dataset)} Val (test isolated).")

    # 3. V1 path
    v1_ckpt_path = REPO_ROOT / "results" / "photon_v1" / "full_training" / "best_model.pt"
    if not v1_ckpt_path.exists():
        v1_ckpt_path = REPO_ROOT / "checkpoints" / "v1_diffusion" / "best_diffusion.pt"

    # 4. Frozen V1 baseline (no training needed)
    print("\n========================================================")
    print(" LOADING FROZEN V1 BASELINE")
    print("========================================================")
    v1_denoiser = LightweightDenoiser(latent_dim=64, hidden_dim=128, num_blocks=2).to(device)
    v1_denoiser.load_state_dict(torch.load(v1_ckpt_path, map_location=device))
    v1_denoiser.eval()
    v1_physics_head = LatentPhysicsHead(latent_dim=64, hidden_dim=32).to(device)
    v1_physics_head.eval()
    v1_physics_loss = RadarPhysicsLoss(
        dt=DT, velocity_sign=1, gap_alpha=0.0, physics_head=v1_physics_head,
    ).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=50, beta_schedule="linear").to(device)

    # 5. Train V2.3 (fixed physics, gap_alpha=0)
    v23_result = train_v2_model(
        train_loader=train_subset_loader,
        val_loader=val_loader,
        encoder=encoder,
        v1_ckpt_path=v1_ckpt_path,
        device=device,
        save_dir=REPO_ROOT / "checkpoints" / "v2_physics" / "v24_experiment" / "v23_fixed",
        gap_alpha=0.0,
        lambda_phys=0.01,
        model_label="V2.3_Fixed",
    )

    # 6. Train V2.4 (gap-aware physics, gap_alpha=0.5)
    v24_result = train_v2_model(
        train_loader=train_subset_loader,
        val_loader=val_loader,
        encoder=encoder,
        v1_ckpt_path=v1_ckpt_path,
        device=device,
        save_dir=REPO_ROOT / "checkpoints" / "v2_physics" / "v24_experiment" / "v24_gap",
        gap_alpha=0.5,
        lambda_phys=0.01,
        model_label="V2.4_GapAware",
    )

    # 7. Evaluate all three models at 20%, 40%, 50% dropout
    all_rows = []
    models = {
        "V1": (v1_denoiser, v1_physics_head, v1_physics_loss),
        "V2.3_Fixed": (v23_result["denoiser"], v23_result["physics_head"], v23_result["physics_loss_fn"]),
        "V2.4_GapAware": (v24_result["denoiser"], v24_result["physics_head"], v24_result["physics_loss_fn"]),
    }

    print("\n========================================================")
    print(" MULTI-DROPOUT EVALUATION")
    print("========================================================")

    for p_val in DROPOUT_LEVELS:
        set_seed(42)
        corr_p = RadarLatentCorruption({"enabled": True, "frame_dropout": {"enabled": True, "probability": p_val}})

        for model_name, (den, ph, pl) in models.items():
            set_seed(42)  # Ensure identical corruption masks
            res = evaluate_model(
                denoiser=den, physics_head=ph, scheduler=scheduler,
                encoder=encoder, physics_loss_module=pl,
                data_loader=val_loader, corr_op=corr_p, device=device,
            )

            row = {
                "model": model_name,
                "dropout_p": p_val,
                "macro_f1": res["macro_f1"],
                "accuracy": res["accuracy"],
                "auroc": res["auroc"],
                "missing_mse": res["missing_mse"],
                "range_mae": res["range_mae"],
                "velocity_mae": res["velocity_mae"],
                "kinematic_residual": res["kinematic_residual"],
                "f1_cyclist": res.get("f1_cyclist", 0.0),
                "f1_empty": res.get("f1_empty", 0.0),
                "f1_pedestrian": res.get("f1_pedestrian", 0.0),
                "f1_vehicle": res.get("f1_vehicle", 0.0),
            }
            all_rows.append(row)

            print(
                f"[{model_name:15s} | p={int(p_val*100):02d}%] "
                f"F1: {res['macro_f1']:.4f}, Cyclist: {res.get('f1_cyclist', 0):.4f}, "
                f"Miss MSE: {res['missing_mse']:.4f}, R_MAE: {res['range_mae']:.3f}m, "
                f"Kin: {res['kinematic_residual']:.3f}"
            )

    # 8. Save CSV
    csv_path = results_dir / "v2_4_gap_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"\n[V2.4] Saved CSV: {csv_path}")

    # 9. Generate Report
    report_path = results_dir / "V2_4_GAP_AWARE_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PhotonShield AI -- Phase V2.4 Gap-Aware Physics Report\n\n")

        # Checkpoint selection audit
        f.write("## Checkpoint Selection Audit\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Tiny ablation checkpoint selection uses Validation Missing-Frame MSE.**\n")
        f.write("> Full V2.3 training uses Validation Macro-F1 with MSE tie-breaker.\n")
        f.write("> This tiny experiment follows the tiny ablation convention (MSE-based selection).\n\n")

        f.write(f"- V2.3 Fixed best epoch: {v23_result['metrics'].get('best_epoch', 'N/A')}\n")
        f.write(f"- V2.4 Gap-Aware best epoch: {v24_result['metrics'].get('best_epoch', 'N/A')}\n\n")

        # Gap weight formula
        f.write("## Gap-Aware Weighting Formula\n\n")
        f.write("$$w_{\\text{gap}}(t) = \\frac{1}{1 + \\alpha \\cdot \\text{gap\\_length}(t)}$$\n\n")
        f.write("- $\\alpha = 0.5$ (configurable)\n")
        f.write("- $\\text{gap\\_length}(t)$ = contiguous missing frames spanning transition $t \\to t+1$\n")
        f.write("- Short gaps: $w \\approx 1$ (strong physics)\n")
        f.write("- Long gaps: $w \\to 0$ (relaxed physics)\n\n")

        # Comparison table
        f.write("## Multi-Dropout Comparison\n\n")
        f.write("| Dropout | Model | Macro-F1 | Cyclist F1 | Miss MSE | Range MAE | Vel MAE | Kin Residual |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for p_val in DROPOUT_LEVELS:
            for model_name in ["V1", "V2.3_Fixed", "V2.4_GapAware"]:
                r = [x for x in all_rows if x["model"] == model_name and x["dropout_p"] == p_val][0]
                bold = "**" if model_name == "V2.4_GapAware" else ""
                f.write(
                    f"| {int(p_val*100)}% | {bold}{model_name}{bold} | "
                    f"`{r['macro_f1']:.4f}` | `{r['f1_cyclist']:.4f}` | "
                    f"`{r['missing_mse']:.4f}` | `{r['range_mae']:.3f}m` | "
                    f"`{r['velocity_mae']:.3f}m/s` | `{r['kinematic_residual']:.3f}` |\n"
                )
            f.write("| | | | | | | | |\n")

        # 50% deep analysis
        f.write("\n## 50% Dropout Deep Analysis\n\n")
        f.write("| Class | V1 F1 | V2.3 F1 | V2.4 F1 | V2.4 vs V2.3 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        r_v1_50 = [x for x in all_rows if x["model"] == "V1" and x["dropout_p"] == 0.50][0]
        r_v23_50 = [x for x in all_rows if x["model"] == "V2.3_Fixed" and x["dropout_p"] == 0.50][0]
        r_v24_50 = [x for x in all_rows if x["model"] == "V2.4_GapAware" and x["dropout_p"] == 0.50][0]
        for c_name in CLASS_NAMES:
            key = f"f1_{c_name.lower()}"
            delta = r_v24_50[key] - r_v23_50[key]
            f.write(
                f"| **{c_name}** | `{r_v1_50[key]:.4f}` | `{r_v23_50[key]:.4f}` | "
                f"`{r_v24_50[key]:.4f}` | `{delta:+.4f}` |\n"
            )

        # Success criteria evaluation
        f.write("\n## Success Criteria Evaluation\n\n")
        c1 = r_v24_50["macro_f1"] > r_v23_50["macro_f1"]
        c2 = r_v24_50["f1_cyclist"] > r_v23_50["f1_cyclist"]
        c3 = r_v24_50["kinematic_residual"] < r_v1_50["kinematic_residual"]
        c4 = r_v24_50["missing_mse"] < r_v23_50["missing_mse"] * 1.2  # not substantially worse

        f.write(f"1. 50% Macro-F1 V2.4 > V2.3: **{'PASS' if c1 else 'FAIL'}** ")
        f.write(f"({r_v24_50['macro_f1']:.4f} vs {r_v23_50['macro_f1']:.4f})\n")
        f.write(f"2. 50% Cyclist F1 V2.4 > V2.3: **{'PASS' if c2 else 'FAIL'}** ")
        f.write(f"({r_v24_50['f1_cyclist']:.4f} vs {r_v23_50['f1_cyclist']:.4f})\n")
        f.write(f"3. Physics consistency better than V1: **{'PASS' if c3 else 'FAIL'}** ")
        f.write(f"(Kin: {r_v24_50['kinematic_residual']:.3f} vs {r_v1_50['kinematic_residual']:.3f})\n")
        f.write(f"4. Missing MSE not substantially degraded: **{'PASS' if c4 else 'FAIL'}** ")
        f.write(f"({r_v24_50['missing_mse']:.4f} vs {r_v23_50['missing_mse']:.4f})\n\n")

        status = "PROMISING" if (c1 and c2 and c3) else "FAILED"
        f.write(f"## FINAL STATUS: **{status}**\n")

    print(f"[V2.4] Saved report: {report_path}")
    print(f"\n{'='*60}")
    print(f" V2.4 GAP-AWARE TINY EXPERIMENT COMPLETE")
    print(f" FINAL STATUS: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_v2_4_gap_experiment()
