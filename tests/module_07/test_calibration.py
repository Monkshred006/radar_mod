"""Tests for Temperature Scaling probability calibration."""

import pytest
import torch
from module_07_decision.calibration import TemperatureScaler


class TestTemperatureScaler:
    def test_temperature_scaling_forward(self):
        scaler = TemperatureScaler(initial_temperature=2.0)
        logits = torch.tensor([[4.0, 2.0], [0.0, -2.0]])
        calibrated = scaler(logits)
        assert torch.allclose(calibrated, logits / 2.0)

    def test_fit_validation(self):
        scaler = TemperatureScaler(initial_temperature=5.0)
        val_logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0], [8.0, -8.0], [-8.0, 8.0]])
        val_labels = torch.tensor([0, 1, 0, 1])

        t_opt = scaler.fit_validation(val_logits, val_labels, is_binary=False)
        assert t_opt > 0.0
        assert torch.isfinite(torch.tensor(t_opt))
