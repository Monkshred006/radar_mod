"""Tests for ActionSpec and ActionEncoder."""

import pytest
import torch

from module_08_pinn_rl.action import ActionEncoder, ActionSpec


class TestAction:
    def test_discrete_encoding(self):
        spec = ActionSpec.discrete(4)
        encoder = ActionEncoder(spec)
        assert encoder.encoded_dim == 4

        enc0 = encoder.encode(0)
        assert torch.equal(enc0, torch.tensor([1.0, 0.0, 0.0, 0.0]))
        assert encoder.decode(enc0) == 0

        enc3 = encoder.encode(3)
        assert torch.equal(enc3, torch.tensor([0.0, 0.0, 0.0, 1.0]))
        assert encoder.decode(enc3) == 3

    def test_discrete_out_of_range_raises(self):
        spec = ActionSpec.discrete(4)
        encoder = ActionEncoder(spec)
        with pytest.raises(ValueError):
            encoder.encode(4)
        with pytest.raises(ValueError):
            encoder.encode(-1)

    def test_continuous_encoding(self):
        spec = ActionSpec.continuous(dim=2)
        encoder = ActionEncoder(spec)
        assert encoder.encoded_dim == 2

        act = [0.5, -0.2]
        enc = encoder.encode(act)
        assert enc.shape == (2,)
        assert enc[0].item() == pytest.approx(0.5)
        assert enc[1].item() == pytest.approx(-0.2)
