"""CLI: python -m module_05_training.train --config <path>

Runs the PhotonShield FP32 training pipeline from a YAML or JSON config file.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhotonShield AI — Module 5 FP32 Training"
    )
    parser.add_argument("--config", required=True, help="Path to YAML or JSON training config")
    parser.add_argument(
        "--variant",
        default="full_mamba_hybrid",
        choices=["baseline_mamba", "sensor_interaction_only", "full_mamba_hybrid"],
        help="Ablation variant to train (default: full_mamba_hybrid)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    # Load config dict
    suffix = cfg_path.suffix.lower()
    if suffix == ".json":
        with open(cfg_path, encoding="utf-8") as f:
            cfg_dict = json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(cfg_path, encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)
        except ImportError:
            print("[ERROR] PyYAML not installed. Use: pip install pyyaml", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[ERROR] Unsupported config format: {suffix}", file=sys.stderr)
        sys.exit(1)

    # Build TrainingConfig
    from module_05_training.config import TrainingConfig
    from module_04_mamba_hybrid.config import MambaHybridConfig

    train_cfg_dict = cfg_dict.get("training", cfg_dict)
    model_cfg_dict = cfg_dict.get("model", {})

    train_cfg = TrainingConfig(**{
        k: v for k, v in train_cfg_dict.items()
        if k in TrainingConfig.__dataclass_fields__
    })
    model_cfg = MambaHybridConfig(**{
        k: v for k, v in model_cfg_dict.items()
        if k in MambaHybridConfig.__dataclass_fields__
    })

    print("=" * 60)
    print("PhotonShield AI — Module 5 FP32 Training")
    print("=" * 60)
    print(f"  Config:   {cfg_path}")
    print(f"  Variant:  {args.variant}")
    print(f"  Device:   {train_cfg.device}")
    print(f"  Seed:     {train_cfg.random_seed}")
    print(f"  Epochs:   {train_cfg.epochs}")
    print(f"  LR:       {train_cfg.learning_rate}")
    print(f"  FP32:     {'YES (mixed_precision=False)' if not train_cfg.mixed_precision else 'WARNING: mixed_precision=True'}")
    print("=" * 60)

    # Build synthetic cache for demo (replace with real data pipeline when available)
    from module_05_training.dataset import make_synthetic_scene_cache
    print("\n[NOTE] No real dataset path found. Running with synthetic data for verification.")
    train_cache = make_synthetic_scene_cache(num_scenes=6, frames_per_scene=60)
    val_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=40)
    test_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=40)

    from module_05_training.experiment import ExperimentRunner
    runner = ExperimentRunner(
        train_config=train_cfg,
        train_cache=train_cache,
        val_cache=val_cache,
        test_cache=test_cache,
        model_config=model_cfg,
        report_dir=str(Path(train_cfg.log_dir) / "reports"),
    )
    report = runner.run(variant=args.variant)
    print(f"\n[Done] Best val metric: {report['training_summary'].get('best_val_metric')}")


if __name__ == "__main__":
    main()
