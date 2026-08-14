"""Tests for threshold sweeping analysis on validation set."""

import pytest
import numpy as np
from module_07_decision.thresholds import analyze_validation_thresholds


class TestThresholdAnalysis:
    def test_threshold_sweep(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
        y_probs = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.3, 0.85, 0.15])

        report = analyze_validation_thresholds(y_true, y_probs)

        assert "thresholds" in report
        assert "precisions" in report
        assert "best_f1_threshold" in report
        assert report["best_f1_score"] > 0.0
