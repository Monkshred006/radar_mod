"""Tests for Environmental Assessment Head."""

import pytest
import torch
from module_07_decision.environmental_head import EnvironmentalHead
from module_07_decision.config import DecisionModelConfig


class TestEnvironmentalHead:
    def test_regression_mode_shape(self):
        head = EnvironmentalHead(d_model=32, mode="regression", num_outputs=3)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 3)

    def test_classification_mode_shape(self):
        head = EnvironmentalHead(d_model=32, mode="classification", num_classes=4)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 4)

    def test_from_config(self):
        cfg = DecisionModelConfig(d_model=64, environment_mode="regression", num_environment_outputs=5)
        head = EnvironmentalHead.from_config(cfg)
        assert head.out_dim == 5
