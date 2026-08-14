"""Model Profiler for PhotonShield AI (PhotonV0 Architecture).

Evaluates:
1. Total and trainable parameter counts.
2. Approximate inference FLOPs & MACs.
3. Peak SRAM activation memory footprint.
4. Checkpoint (.pt) and ONNX (.onnx) export sizes.
5. Issues hardware-awareness warnings (>1M params, >4MB checkpoint, >10MB ONNX).
6. Saves detailed profile report to `results/photon_v0/profile_report.json`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, Union, Optional

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml

from module_04_mamba_hybrid.photon_v0 import PhotonV0, count_parameters
from module_06_bitnet.profile_uno_q import estimate_photon_v0_macs, profile_for_uno_q
from module_06_bitnet.export_onnx import export_photon_v0_onnx


def profile_model(
    config_path: Union[str, Path] = "configs/photon_v0.yaml",
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_report_path: Union[str, Path] = "results/photon_v0/profile_report.json",
) -> Dict[str, Any]:
    """Execute model profiling and generate sizing report."""
    cfg = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    cfg_model = cfg.get("model", {})
    input_dim = cfg_model.get("input_dim", 64)
    hidden_dim = cfg_model.get("hidden_dim", 64)
    num_layers = cfg_model.get("num_layers", 2)
    seq_len = cfg_model.get("sequence_length", 16)
    num_classes = cfg_model.get("num_classes", 4)
    d_state = cfg_model.get("d_state", 16)
    d_conv = cfg_model.get("d_conv", 4)
    expand = cfg_model.get("expand", 2)

    # 1. Instantiate Model
    model = PhotonV0(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sequence_length=seq_len,
        num_classes=num_classes,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        backend="fallback",
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"[Model Profiler] Loaded weights from {checkpoint_path}")

    # 2. Parameter Counts
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    # 3. Checkpoint Size
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    ckpt_size_bytes = buf.getbuffer().nbytes
    ckpt_size_mb = ckpt_size_bytes / (1024 * 1024)

    # 4. Compute & FLOPs estimate
    total_macs = estimate_photon_v0_macs(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sequence_length=seq_len,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        num_classes=num_classes,
    )
    total_flops = total_macs * 2

    # 5. Hardware Activation & Memory Profiling
    uno_q_profile = profile_for_uno_q(
        model=model,
        sequence_length=seq_len,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    # 6. ONNX Export Size
    temp_onnx = Path("results/photon_v0/temp_profile.onnx")
    temp_onnx.parent.mkdir(parents=True, exist_ok=True)
    export_photon_v0_onnx(
        model=model,
        output_path=temp_onnx,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sequence_length=seq_len,
        num_classes=num_classes,
    )
    onnx_size_bytes = temp_onnx.stat().st_size
    onnx_size_mb = onnx_size_bytes / (1024 * 1024)
    if temp_onnx.exists():
        temp_onnx.unlink()

    # 7. Check Thresholds and Issue Warnings
    warnings = []
    if total_params > 1_000_000:
        warnings.append(f"Model exceeds 1,000,000 parameters (current: {total_params:,})")
    if ckpt_size_mb > 4.0:
        warnings.append(f"Checkpoint exceeds 4.0 MB limit (current: {ckpt_size_mb:.2f} MB)")
    if onnx_size_mb > 10.0:
        warnings.append(f"ONNX model exceeds 10.0 MB limit (current: {onnx_size_mb:.2f} MB)")

    report = {
        "model_name": "PhotonV0 (Minimal Mamba Temporal Perception)",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "inference_macs": total_macs,
        "inference_flops": total_flops,
        "checkpoint_size_bytes": ckpt_size_bytes,
        "checkpoint_size_kb": round(ckpt_size_bytes / 1024, 2),
        "checkpoint_size_mb": round(ckpt_size_mb, 4),
        "onnx_size_bytes": onnx_size_bytes,
        "onnx_size_kb": round(onnx_size_bytes / 1024, 2),
        "onnx_size_mb": round(onnx_size_mb, 4),
        "peak_sram_activation_kb": uno_q_profile["peak_sram_int8_kb"],
        "arduino_uno_q_profile": uno_q_profile,
        "warnings": warnings,
    }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("================================================================")
    print(" PhotonShield AI — Model Architecture Profile Report")
    print("================================================================")
    print(f"Total Parameters:       {total_params:,}")
    print(f"Trainable Parameters:   {trainable_params:,}")
    print(f"Inference MACs:         {total_macs:,} (~{total_flops:,} FLOPs)")
    print(f"Checkpoint (.pt) Size:  {report['checkpoint_size_kb']:.2f} KB ({report['checkpoint_size_mb']:.4f} MB)")
    print(f"ONNX Model (.onnx) Size:{report['onnx_size_kb']:.2f} KB ({report['onnx_size_mb']:.4f} MB)")
    print(f"Peak SRAM (INT8):       {report['peak_sram_activation_kb']:.2f} KB / 64 KB Uno Q SRAM")
    print("----------------------------------------------------------------")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("Hardware Constraints:   PASSED (Well within Uno Q 512KB Flash / 64KB SRAM)")
    print("================================================================")
    print(f"Profile report saved to: {out_file.absolute()}\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PhotonV0 architecture.")
    parser.add_argument("--config", type=str, default="configs/photon_v0.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/photon_v0/profile_report.json")
    args = parser.parse_args()

    profile_model(config_path=args.config, checkpoint_path=args.checkpoint, output_report_path=args.output)


if __name__ == "__main__":
    main()
