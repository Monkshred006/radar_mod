"""Diagnostic utilities and evaluation metrics for adaptive compute schedulers."""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from module_07_adaptive_compute.action_space import ACTIONS, ACTION_TO_IDX


def compute_policy_metrics(
    oracle_actions: List[int],
    predicted_actions: List[int],
    oracle_objectives: List[float],
    selected_objectives: List[float],
) -> Dict[str, Any]:
    """Calculate agreement accuracy, regret, oracle gap, and action distribution."""
    y_true = np.array(oracle_actions)
    y_pred = np.array(predicted_actions)
    j_orc = np.array(oracle_objectives)
    j_sel = np.array(selected_objectives)

    accuracy = float(accuracy_score(y_true, y_pred))
    agreement_pct = accuracy * 100.0

    # Regret / Oracle gap
    regret = j_sel - j_orc
    mean_regret = float(np.mean(regret))
    median_regret = float(np.median(regret))
    max_regret = float(np.max(regret))

    # Compute step statistics
    mean_steps = float(np.mean(y_pred))
    median_steps = float(np.median(y_pred))
    p95_steps = float(np.percentile(y_pred, 95))
    compute_reduction = (1.0 - (mean_steps / 50.0)) * 100.0

    # 4x4 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=ACTIONS)

    # Action distribution
    dist = {f"P_{N}steps": float(np.mean(y_pred == N)) for N in ACTIONS}

    return {
        "accuracy": accuracy,
        "agreement_pct": agreement_pct,
        "mean_regret": mean_regret,
        "median_regret": median_regret,
        "max_regret": max_regret,
        "mean_steps": mean_steps,
        "median_steps": median_steps,
        "p95_steps": p95_steps,
        "compute_reduction_pct": compute_reduction,
        "confusion_matrix": cm,
        "action_distribution": dist,
    }
