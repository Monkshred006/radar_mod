"""Tests for Anomaly Detection Head."""

import pytest
import torch
from module_07_decision.anomaly_head import AnomalyHead
from module_07_decision.config import DecisionModelConfig


class TestAnomalyHead:
    def test_forward_output_shape(self):
        head = AnomalyHead(d_model=32)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 1)
        assert torch.isfinite(out).all()

    def test_from_config(self):
        cfg = DecisionModelConfig(d_model=64, anomaly_hidden_dim=16)
        head = AnomalyHead.from_config(cfg)
        assert head.d_model == 64

    def test_gradient_flow(self):
        head = AnomalyHead(d_model=16)
        x = torch.randn(2, 16, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
