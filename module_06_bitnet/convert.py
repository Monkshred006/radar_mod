"""CLI: python -m module_06_bitnet.convert --checkpoint <fp32_checkpoint> --output <output_checkpoint>

Converts a trained FP32 Module 5 model into a BitNet-compatible mixed-precision model.
Prints layer inspection report, quantization status, scaling parameters, and conversion statistics.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.model_conversion import convert_fp32_to_bitnet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhotonShield AI — Module 6 BitNet Model Conversion"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained FP32 checkpoint file (.pt)")
    parser.add_argument("--output", default="checkpoints/bitnet/bitnet_converted.pt", help="Destination path for BitNet checkpoint")
    parser.add_argument("--scaling", default="mean_abs", choices=["mean_abs", "max_abs"], help="Weight scaling method")
    args = parser.parse_args()

    cfg = BitNetConfig(scaling_method=args.scaling)
    print("=" * 70)
    print("PhotonShield AI — Module 6 BitNet Model Conversion")
    print("=" * 70)
    print(f"  Source Checkpoint: {args.checkpoint}")
    print(f"  Output Checkpoint: {args.output}")
    print(f"  Scaling Method:    {args.scaling}")
    print("=" * 70)

    try:
        engine_bitnet, head_bitnet, report = convert_fp32_to_bitnet(
            fp32_checkpoint_path=args.checkpoint,
            bitnet_config=cfg,
            output_checkpoint_path=args.output,
        )
        print("\n[Layer Inspection Report]")
        print(report["layer_inspection_report"]["formatted_table"])
        print(f"\n[Conversion Complete] BitNet checkpoint saved: {args.output}")
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
