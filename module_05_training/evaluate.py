"""CLI: python -m module_05_training.evaluate --checkpoint <path>

Loads a saved checkpoint and runs test-set evaluation.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhotonShield AI — Module 5 FP32 Evaluation"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    import torch
    from module_04_mamba_hybrid.config import MambaHybridConfig
    from module_04_mamba_hybrid.engine import PhotonMambaHybrid
    from module_04_mamba_hybrid.heads import ClassificationHead, RegressionHead
    from module_05_training.config import TrainingConfig
    from module_05_training.checkpointing import load_checkpoint
    from module_05_training.evaluator import Evaluator
    from module_05_training.dataset import make_synthetic_scene_cache, PhotonShieldDataset, collate_module3
    from module_05_training.target_adapter import get_target_adapter
    from torch.utils.data import DataLoader

    # Load checkpoint to read configs
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[Evaluator] Loaded checkpoint from epoch {raw.get('epoch')}")

    # Reconstruct configs from checkpoint data
    train_cfg_data = raw.get("training_config", {})
    model_cfg_data = raw.get("model_config", {})

    train_cfg = TrainingConfig(**{
        k: v for k, v in train_cfg_data.items()
        if k in TrainingConfig.__dataclass_fields__
    })
    model_cfg = MambaHybridConfig(**{
        k: v for k, v in model_cfg_data.items()
        if k in MambaHybridConfig.__dataclass_fields__
    })

    # Rebuild model
    engine = PhotonMambaHybrid(model_cfg)
    from module_04_mamba_hybrid.config import TaskHeadConfig
    if train_cfg.target_type == "classification":
        head_cfg = TaskHeadConfig(head_type="classification", num_classes=train_cfg.num_classes)
        head = ClassificationHead(model_cfg.d_model, head_cfg)
    else:
        head_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=train_cfg.num_regression_outputs)
        head = RegressionHead(model_cfg.d_model, head_cfg)

    combined = torch.nn.ModuleList([engine, head])
    load_checkpoint(str(ckpt_path), combined)
    print("[Evaluator] Model weights restored.")

    # Use synthetic test data (replace with real test loader when available)
    print("[NOTE] Using synthetic test data.")
    test_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=40)
    adapter = get_target_adapter(train_cfg)
    test_ds = PhotonShieldDataset(test_cache, adapter,
                                   window_len=train_cfg.sequence_length,
                                   window_stride=train_cfg.sequence_stride)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size,
                              shuffle=False, collate_fn=collate_module3)

    evaluator = Evaluator(train_cfg)
    results = evaluator.evaluate(engine, head, test_loader)

    print("\n[Evaluation Results]")
    print(f"  Test Loss:         {results['loss']:.5f}")
    print(f"  Sample Count:      {results['sample_count']}")
    print(f"  Inference Time:    {results['inference_time_s']:.3f}s")
    print(f"  Per-Sample (ms):   {results['per_sample_latency_ms']:.2f}")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.5f}")


if __name__ == "__main__":
    main()
