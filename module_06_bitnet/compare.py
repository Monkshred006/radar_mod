"""CLI: python -m module_06_bitnet.compare --fp32 <fp32_checkpoint> --bitnet <bitnet_checkpoint>

Runs comparison experiment matrix comparing FP32 Baseline vs Direct-Ternary PTQ vs BitNet-Style QAT.
"""

from __future__ import annotations
import argparse
import sys

from module_05_training.config import TrainingConfig
from module_05_training.dataset import make_synthetic_scene_cache
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.experiment import BitNetExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhotonShield AI — Module 6 Comparison Matrix Generator"
    )
    parser.add_argument("--fp32", required=True, help="Path to reference FP32 checkpoint file")
    parser.add_argument("--output", default="reports/bitnet", help="Output directory for reports")
    args = parser.parse_args()

    train_cfg = TrainingConfig()
    bitnet_cfg = BitNetConfig()

    train_cache = make_synthetic_scene_cache(num_scenes=4, frames_per_scene=40)
    val_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=30)
    test_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=30)

    runner = BitNetExperimentRunner(
        fp32_checkpoint_path=args.fp32,
        train_config=train_cfg,
        train_cache=train_cache,
        val_cache=val_cache,
        test_cache=test_cache,
        bitnet_config=bitnet_cfg,
        output_dir=args.output,
    )

    report = runner.run_comparison_experiment()

    print("\n" + "=" * 70)
    print("PhotonShield AI — FP32 vs PTQ vs BitNet QAT Comparison Matrix")
    print("=" * 70)
    print(report["markdown_table"])
    print("\n[Reports Saved]")
    print(f"  JSON:     {args.output}/comparison_matrix.json")
    print(f"  Markdown: {args.output}/comparison_matrix.md")


if __name__ == "__main__":
    main()
