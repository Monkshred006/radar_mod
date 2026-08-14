"""Module 6 Evaluation Integration with Module 5.

Guarantees fair evaluation by reusing Module 5's exact test set, dataset split,
target adapter, and Evaluator instance.
"""

from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module_05_training.config import TrainingConfig
from module_05_training.evaluator import Evaluator


def evaluate_bitnet_model(
    engine: nn.Module,
    head: nn.Module,
    train_config: TrainingConfig,
    test_loader: DataLoader,
    variant_label: str = "BitNet-Style QAT",
) -> Dict[str, Any]:
    """Evaluate a BitNet model using Module 5's Evaluator.

    Args:
        engine: BitNet Module 4 engine.
        head: BitNet task head.
        train_config: TrainingConfig.
        test_loader: Test DataLoader.
        variant_label: Label for experiment report ('FP32 Baseline', 'Direct-Ternary PTQ', 'BitNet-Style QAT').

    Returns:
        Evaluation metrics dictionary.
    """
    evaluator = Evaluator(train_config)
    results = evaluator.evaluate(engine, head, test_loader)
    results["model_variant"] = variant_label
    return results
