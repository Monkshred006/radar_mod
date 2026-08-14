"""Post-Training Quantization (Direct-Ternary PTQ) Baseline Evaluator.

Evaluates a direct-converted BitNet model (without QAT fine-tuning) on the held-out
test set to establish the Direct-Ternary PTQ baseline for scientific comparison.
"""

from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module_05_training.config import TrainingConfig
from module_05_training.evaluator import Evaluator


def evaluate_ptq_baseline(
    engine: nn.Module,
    head: nn.Module,
    train_config: TrainingConfig,
    test_loader: DataLoader,
) -> Dict[str, Any]:
    """Run Direct-Ternary PTQ evaluation on test DataLoader.

    Args:
        engine: Converted BitNet Module 4 engine.
        head: Converted BitNet task head.
        train_config: TrainingConfig specifying loss and target options.
        test_loader: Held-out test set DataLoader from Module 5.

    Returns:
        Dict of evaluation metrics and sample counts for Direct-Ternary PTQ.
    """
    evaluator = Evaluator(train_config)
    results = evaluator.evaluate(engine, head, test_loader)
    results["model_variant"] = "Direct-Ternary PTQ"
    return results
