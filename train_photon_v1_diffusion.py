"""PhotonShield AI - Phase V1.0 Latent Diffusion Training & Evaluation Pipeline.

Executes conditional latent diffusion training to reconstruct corrupted temporal radar latents.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from module_03_sensor_fusion.radical_adapter import RaDICaLDatasetAdapter
from module_05_latent_diffusion.latent_diffusion import LatentDiffusionModel
from module_05_latent_diffusion.trainer import DiffusionTrainer
from module_05_latent_diffusion.evaluator import DiffusionEvaluator


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhotonShield AI Phase V1.0 Latent Diffusion Model.")
    parser.add_argument("--config", type=str, default="configs/photon_v1_diffusion.yaml", help="Path to config yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg_train = config.get("training", {})
    cfg_data = config.get("dataset", {})
    cfg_v0 = config.get("v0", {})
    cfg_diff = config.get("diffusion", {})
    cfg_corr = config.get("corruption", {})

    seed = int(cfg_train.get("seed", 42))
    set_seed(seed)

    # Device
    dev_str = cfg_train.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() and dev_str == "auto" else dev_str)
    print(f"[PhotonShield AI V1.0] Device selected: {device}")

    # 1. Load Dataset with exact frozen splits
    adapter = RaDICaLDatasetAdapter(
        data_path=cfg_data.get("data_path", "C:/Users/worka/research/photonpinn/data/radical"),
        splits_dir=cfg_data.get("splits_dir", "C:/Users/worka/research/photonpinn/data/radical/splits"),
        sequence_length=cfg_data.get("sequence_length", 16),
        feature_dim=cfg_data.get("feature_dim", 64),
        num_classes=cfg_data.get("num_classes", 4),
        normalization=cfg_data.get("normalization", "db"),
        seed=seed,
        synthetic_fallback=False,
    )

    batch_size = int(cfg_train.get("batch_size", 16))
    train_loader, val_loader, test_loader = adapter.get_dataloaders(batch_size=batch_size)
    print(f"[PhotonShield AI V1.0] RaDICaL Splits: {len(train_loader.dataset)} Train, {len(val_loader.dataset)} Val, {len(test_loader.dataset)} Test")

    # 2. Build Latent Diffusion Model with Frozen V0 Encoder
    v0_ckpt = cfg_v0.get("checkpoint", "checkpoints/v0_frozen/best_model.pt")
    model = LatentDiffusionModel(
        v0_checkpoint_path=v0_ckpt,
        latent_dim=int(cfg_diff.get("latent_dim", 64)),
        hidden_dim=int(cfg_diff.get("hidden_dim", 128)),
        num_blocks=int(cfg_diff.get("num_blocks", 2)),
        timesteps=int(cfg_diff.get("timesteps", 50)),
        corruption_config=cfg_corr,
    ).to(device)

    denoiser_params = model.denoiser.count_parameters()
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print("================================================================")
    print(f" PhotonShield AI V1.0 Latent Diffusion Model Initialized")
    print(f" - Frozen V0 Encoder Parameters: {encoder_params:,} (requires_grad = False)")
    print(f" - Trainable Denoiser Parameters: {denoiser_params:,}")
    print(f" - Active Corruption: Frame Dropout (p={cfg_corr.get('frame_dropout', {}).get('probability', 0.20):.2f})")
    print("================================================================")

    # 3. Train Diffusion Denoiser
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )
    train_summary = trainer.train()
    train_summary["epochs"] = config["training"]["epochs"]

    # 4. Evaluate on Test Set using Best Checkpoint
    best_ckpt_path = Path("checkpoints/v1_diffusion/best_diffusion.pt")
    if best_ckpt_path.exists():
        model.denoiser.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"[PhotonShield AI V1.0] Loaded best denoiser weights from '{best_ckpt_path}'")

    results_dir = Path("results/photon_v1")
    evaluator = DiffusionEvaluator(
        model=model,
        test_loader=test_loader,
        results_dir=results_dir,
        device=device,
    )
    eval_results = evaluator.evaluate_test_set(num_inference_steps=int(cfg_diff.get("timesteps", 50)))

    # Save summary JSON and Markdown report
    evaluator.generate_v1_report(train_results=train_summary, eval_results=eval_results, config=config)

    with open(results_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_summary, "eval": eval_results}, f, indent=2)

    status_str = "PASS" if eval_results["gate_passed"] else "FAIL"
    next_stage_str = "V1.1 JOINT PERCEPTION" if eval_results["gate_passed"] else "INVESTIGATE DIFFUSION"

    print("\n================================================================")
    print("V1.0 LATENT DIFFUSION COMPLETE")
    print()
    print(f"Corruption:")
    print(f"Temporal frame dropout, p={cfg_corr.get('frame_dropout', {}).get('probability', 0.20):.2f}")
    print()
    print(f"Baseline corrupted latent MSE:")
    print(f"{eval_results['corrupted_latent_mse']:.6f}")
    print()
    print(f"Reconstructed latent MSE:")
    print(f"{eval_results['reconstructed_latent_mse']:.6f}")
    print()
    print(f"Improvement:")
    print(f"{eval_results['improvement_percentage']:.2f} %")
    print()
    print(f"Best validation diffusion loss:")
    print(f"{train_summary['best_val_rec_mse']:.6f}")
    print()
    print(f"Peak VRAM:")
    print(f"{train_summary['peak_vram_gb']:.4f} GB")
    print()
    print(f"Training time:")
    print(f"{train_summary['total_time_sec']:.2f} seconds")
    print()
    print(f"V1.0 STATUS:")
    print(f"{status_str}")
    print()
    print(f"NEXT:")
    print(f"{next_stage_str}")
    print("================================================================")


if __name__ == "__main__":
    main()
