"""Tests for reproducibility utilities."""

import pytest
import torch
import numpy as np
import random

from module_05_training.reproducibility import set_seed, get_seed_state, restore_seed_state


class TestSetSeed:
    def test_same_seed_same_torch_output(self):
        set_seed(42)
        t1 = torch.randn(5, 5)
        set_seed(42)
        t2 = torch.randn(5, 5)
        assert torch.allclose(t1, t2)

    def test_same_seed_same_numpy_output(self):
        set_seed(42)
        a1 = np.random.randn(10)
        set_seed(42)
        a2 = np.random.randn(10)
        assert np.allclose(a1, a2)

    def test_same_seed_same_python_random(self):
        set_seed(42)
        r1 = [random.random() for _ in range(10)]
        set_seed(42)
        r2 = [random.random() for _ in range(10)]
        assert r1 == r2

    def test_different_seeds_different_outputs(self):
        set_seed(1)
        t1 = torch.randn(5)
        set_seed(2)
        t2 = torch.randn(5)
        assert not torch.allclose(t1, t2)


class TestSeedStateRoundTrip:
    def test_capture_and_restore(self):
        set_seed(99)
        # Advance state
        _ = torch.randn(10)
        _ = np.random.randn(5)
        _ = random.random()

        # Capture current state
        state = get_seed_state()

        # Generate values A
        a_torch = torch.randn(5)
        a_np = np.random.randn(5)
        a_py = [random.random() for _ in range(5)]

        # Restore and regenerate — must be identical
        restore_seed_state(state)
        b_torch = torch.randn(5)
        b_np = np.random.randn(5)
        b_py = [random.random() for _ in range(5)]

        assert torch.allclose(a_torch, b_torch)
        assert np.allclose(a_np, b_np)
        assert a_py == b_py
