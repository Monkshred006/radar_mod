"""Tests for Target Indication Head."""

import pytest
import torch
from module_07_decision.target_head import TargetHead
from module_07_decision.config import DecisionModelConfig


class TestTargetHead:
    def test_forward_output_shape_binary(self):
        head = TargetHead(d_model=32, num_classes=2)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 2)
        assert torch.isfinite(out).all()

    def test_forward_output_shape_multiclass(self):
        head = TargetHead(d_model=64, num_classes=5, hidden_dim=32)
        x = torch.randn(2, 64)
        out = head(x)
        assert out.shape == (2, 5)

    def test_from_config(self):
        cfg = DecisionModelConfig(d_model=128, num_target_classes=4)
        head = TargetHead.from_config(cfg)
        assert head.d_model == 128
        assert head.num_classes == 4

    def test_gradient_flow(self):
        head = TargetHead(d_model=16, num_classes=2)
        x = torch.randn(2, 16, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
