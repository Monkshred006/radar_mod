"""Tests for ternary quantization, STE autograd, and packing subsystem."""

import pytest
import torch
from module_06_bitnet.ternary import (
    TernarySTEFunction,
    round_to_ternary,
    pack_ternary,
    unpack_ternary,
)


class TestTernarySTE:
    def test_forward_values_discrete(self):
        """Unscaled discrete values must strictly be in {-1, 0, 1}."""
        w = torch.randn(10, 10)
        _, _, w_int = round_to_ternary(w)
        unique_vals = set(w_int.numpy().flatten())
        assert unique_vals.issubset({-1, 0, 1})

    def test_ste_backward_gradient_flow(self):
        """STE passes non-zero, finite gradients straight through to master weights."""
        w = torch.randn(4, 4, requires_grad=True)
        scale = torch.tensor(1.0)
        w_quant = TernarySTEFunction.apply(w, scale)
        loss = w_quant.sum()
        loss.backward()

        assert w.grad is not None
        assert torch.isfinite(w.grad).all()
        assert not torch.allclose(w.grad, torch.zeros_like(w.grad))

    def test_boundary_behavior(self):
        """Check values near clipping and zero threshold boundaries."""
        w = torch.tensor([-2.0, -0.6, -0.4, 0.0, 0.4, 0.6, 2.0], requires_grad=True)
        scale = torch.tensor(1.0)
        w_quant = TernarySTEFunction.apply(w, scale)
        loss = w_quant.sum()
        loss.backward()
        assert torch.isfinite(w.grad).all()


class TestTernaryPacking:
    def test_pack_unpack_roundtrip(self):
        """Verify unpack_ternary(pack_ternary(w)) == original ternary tensor."""
        shapes = [(10,), (5, 5), (4, 4, 3), (2, 3, 4, 5)]
        for shape in shapes:
            # Generate random ternary tensor {-1, 0, 1}
            rand = torch.randint(-1, 2, shape, dtype=torch.int8)
            packed = pack_ternary(rand)
            restored = unpack_ternary(packed, rand.shape)
            assert torch.equal(rand, restored), f"Packing failed for shape {shape}"

    def test_packed_size_smaller(self):
        """Packed representation size should be ~1/5th of number of elements."""
        w = torch.randint(-1, 2, (1000,), dtype=torch.int8)
        packed = pack_ternary(w)
        # 1000 elements / 5 per byte = 200 bytes
        assert len(packed) == 200
