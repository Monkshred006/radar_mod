"""Tests for weight and activation scaling functions."""

import pytest
import torch
from module_06_bitnet.scaling import compute_weight_scale, compute_activation_scale


class TestWeightScaling:
    def test_mean_abs_per_tensor(self):
        w = torch.tensor([[1.0, -3.0], [2.0, -4.0]])  # abs values: 1, 3, 2, 4 → mean = 2.5
        scale = compute_weight_scale(w, method="mean_abs", scope="per_tensor")
        assert scale.item() == pytest.approx(2.5)

    def test_max_abs_per_tensor(self):
        w = torch.tensor([[1.0, -3.0], [2.0, -4.0]])
        scale = compute_weight_scale(w, method="max_abs", scope="per_tensor")
        assert scale.item() == pytest.approx(4.0)

    def test_mean_abs_per_channel(self):
        w = torch.tensor([[1.0, -3.0], [2.0, -4.0]])  # row 0 mean = 2.0, row 1 mean = 3.0
        scale = compute_weight_scale(w, method="mean_abs", scope="per_channel")
        assert scale.shape == (2, 1)
        assert scale[0, 0].item() == pytest.approx(2.0)
        assert scale[1, 0].item() == pytest.approx(3.0)

    def test_zero_tensor_eps_clamp(self):
        w = torch.zeros(4, 4)
        scale = compute_weight_scale(w, eps=1e-8)
        assert scale.item() == pytest.approx(1e-8, rel=1e-5)
        assert torch.isfinite(scale)


class TestActivationScaling:
    def test_activation_max_abs(self):
        x = torch.tensor([[-10.0, 5.0], [2.0, 0.0]])
        scale = compute_activation_scale(x, method="max_abs")
        assert scale.item() == pytest.approx(10.0)
