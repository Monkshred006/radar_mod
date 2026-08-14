"""CLI: python -m module_06_bitnet.evaluate --checkpoint <bitnet_checkpoint>

Evaluates a saved BitNet model checkpoint on the test dataset.
"""

from __future__ import annotations
import argparse
import sys
import torch
from torch.utils.data import DataLoader

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import ClassificationHead, RegressionHead
from module_05_training.config import TrainingConfig
from module_05_training.dataset import make_synthetic_scene_cache, PhotonShieldDataset, collate_module3
from module_05_training.target_adapter import get_target_adapter
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.checkpointing import load_bitnet_checkpoint
from module_06_bitnet.layer_replacement import replace_linear_layers
from module_06_bitnet.evaluation import evaluate_bitnet_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhotonShield AI — Module 6 BitNet Model Evaluation"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to BitNet checkpoint file")
    args = parser.parse_args()

    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    m_cfg = MambaHybridConfig(**raw_ckpt["model_config"])
    t_cfg = TrainingConfig(**raw_ckpt["training_config"])
    b_cfg = BitNetConfig(**raw_ckpt["bitnet_config"])

    engine = PhotonMambaHybrid(m_cfg)
    if t_cfg.target_type == "classification":
        h_cfg = TaskHeadConfig(head_type="classification", num_classes=t_cfg.num_classes)
        head = ClassificationHead(m_cfg.d_model, h_cfg)
    else:
        h_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=t_cfg.num_regression_outputs)
        head = RegressionHead(m_cfg.d_model, h_cfg)

    # Convert linear layers to BitLinear before loading state dict
    replace_linear_layers(engine, b_cfg)
    replace_linear_layers(head, b_cfg)

    load_bitnet_checkpoint(args.checkpoint, engine, head)

    test_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=30)
    adapter = get_target_adapter(t_cfg)
    test_ds = PhotonShieldDataset(test_cache, adapter, window_len=t_cfg.sequence_length, window_stride=t_cfg.sequence_stride)
    test_loader = DataLoader(test_ds, batch_size=t_cfg.batch_size, shuffle=False, collate_fn=collate_module3)

    results = evaluate_bitnet_model(engine, head, t_cfg, test_loader)
    print("=" * 60)
    print("PhotonShield AI — BitNet Model Evaluation Results")
    print("=" * 60)
    print(f"  Test Loss:        {results['loss']:.5f}")
    print(f"  Sample Count:     {results['sample_count']}")
    print(f"  Latency (ms):     {results['per_sample_latency_ms']:.2f}")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.5f}")


if __name__ == "__main__":
    main()
