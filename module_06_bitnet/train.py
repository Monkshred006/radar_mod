"""CLI: python -m module_06_bitnet.train --checkpoint <bitnet_checkpoint>

Runs Quantization-Aware Training (QAT) fine-tuning on a converted BitNet model.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from module_05_training.config import TrainingConfig
from module_05_training.dataset import make_synthetic_scene_cache
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.experiment import BitNetExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhotonShield AI — Module 6 BitNet QAT Fine-Tuning"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to reference FP32 or BitNet checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="QAT fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for QAT")
    args = parser.parse_args()

    train_cfg = TrainingConfig(epochs=args.epochs, learning_rate=args.lr)
    bitnet_cfg = BitNetConfig(epochs=args.epochs, learning_rate=args.lr)

    print("=" * 70)
    print("PhotonShield AI — Module 6 BitNet QAT Fine-Tuning")
    print("=" * 70)

    # Demo mode with synthetic data if no real data loader configured
    train_cache = make_synthetic_scene_cache(num_scenes=4, frames_per_scene=50)
    val_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=30)
    test_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=30)

    runner = BitNetExperimentRunner(
        fp32_checkpoint_path=args.checkpoint,
        train_config=train_cfg,
        train_cache=train_cache,
        val_cache=val_cache,
        test_cache=test_cache,
        bitnet_config=bitnet_cfg,
    )
    matrix_report = runner.run_comparison_experiment()
    print("\n[Comparison Matrix Output]")
    print(matrix_report["markdown_table"])


if __name__ == "__main__":
    main()
