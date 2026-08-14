"""Threshold Analysis and Optimization Utilities for Module 7 (Validation Set Only)."""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import torch


def analyze_validation_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold_steps: int = 19,
) -> Dict[str, Any]:
    """Perform threshold sweep analysis on VALIDATION predictions.

    Args:
        y_true: Binary ground truth targets [N] (0 or 1).
        y_probs: Predicted positive class probabilities [N] in [0, 1].
        threshold_steps: Number of threshold evaluation points between 0.05 and 0.95.

    Returns:
        Dict containing threshold grid, precision, recall, f1, FPR, FNR, and optimal F1 threshold.
    """
    thresholds = np.linspace(0.05, 0.95, threshold_steps)
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []
    fpr_list: List[float] = []
    fnr_list: List[float] = []

    best_f1 = -1.0
    best_threshold = 0.5

    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)

    for th in thresholds:
        preds = (y_probs >= th).astype(int)

        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = float(fp / neg_count) if neg_count > 0 else 0.0
        fnr = float(fn / pos_count) if pos_count > 0 else 0.0

        precisions.append(round(prec, 4))
        recalls.append(round(rec, 4))
        f1_scores.append(round(f1, 4))
        fpr_list.append(round(fpr, 4))
        fnr_list.append(round(fnr, 4))

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(th)

    return {
        "thresholds": [round(float(t), 4) for t in thresholds],
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
        "false_positive_rates": fpr_list,
        "false_negative_rates": fnr_list,
        "best_f1_threshold": round(best_threshold, 4),
        "best_f1_score": round(best_f1, 4),
    }
