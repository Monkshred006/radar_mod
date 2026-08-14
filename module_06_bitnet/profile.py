"""CLI: python -m module_06_bitnet.profile --fp32 <fp32_checkpoint> --bitnet <bitnet_checkpoint>

Profiles parameter count, theoretical storage, actual serialized size, CPU/CUDA latency, and memory.
"""

from __future__ import annotations
import argparse
import sys
import torch

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import ClassificationHead, RegressionHead
from module_05_training.config import TrainingConfig
from module_05_training.dataset import make_synthetic_scene_cache
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.checkpointing import load_bitnet_checkpoint
from module_06_bitnet.layer_replacement import replace_linear_layers
from module_06_bitnet.profiling import profile_bitnet_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhotonShield AI — Module 6 BitNet Profiling"
    )
    parser.add_argument("--fp32", help="Path to FP32 checkpoint file")
    parser.add_argument("--bitnet", required=True, help="Path to BitNet checkpoint file")
    args = parser.parse_args()

    raw_ckpt = torch.load(args.bitnet, map_location="cpu", weights_only=False)
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

    replace_linear_layers(engine, b_cfg)
    replace_linear_layers(head, b_cfg)
    load_bitnet_checkpoint(args.bitnet, engine, head)

    sample_cache = make_synthetic_scene_cache(num_scenes=1, frames_per_scene=20)
    sample_window = sample_cache.get_window("synthetic_scene_000", 0, 10)

    prof = profile_bitnet_model(engine, head, sample_window, checkpoint_path=args.bitnet)

    print("=" * 60)
    print("PhotonShield AI — Module 6 BitNet Profiling Report")
    print("=" * 60)
    print(f"  Total Parameters:           {prof['total_params']:,d}")
    print(f"  Ternary Parameters:         {prof['ternary_params']:,d} ({prof['pct_ternary']}%)")
    print(f"  Theoretical Bits/Weight:    {prof['theoretical_bits_per_weight']}")
    print(f"  Theoretical Size (MB):      {prof['theoretical_weight_storage_mb']}")
    print(f"  Actual Checkpoint (MB):     {prof['actual_checkpoint_mb']}")
    print(f"  Mean Latency (ms):          {prof['mean_latency_ms']}")
    print(f"  p50 Latency (ms):           {prof['p50_latency_ms']}")
    print(f"  p95 Latency (ms):           {prof['p95_latency_ms']}")
    print(f"  Throughput (samples/sec):   {prof['throughput_samples_per_sec']}")
    print(f"  Device:                     {prof['device']}")
    print(f"\n[Disclaimer]\n{prof['hardware_disclaimer']}")


if __name__ == "__main__":
    main()
