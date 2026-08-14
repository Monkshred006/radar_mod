"""Unit tests for Mamba Temporal Branch."""

import torch
import pytest
from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.mamba_block import MambaTemporalBranch, FallbackSSMBackend


def test_fallback_ssm_backend_shape_and_gradients():
    config = MambaHybridConfig(d_model=64, mamba_state_dim=16, backend="fallback")
    mamba = FallbackSSMBackend(config)

    x = torch.randn(2, 16, 64, requires_grad=True)
    out = mamba(x)

    assert out.shape == (2, 16, 64)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mamba_temporal_branch_backend():
    config = MambaHybridConfig(d_model=64, backend="fallback")
    branch = MambaTemporalBranch(config)

    assert branch.backend_name == "fallback"

    x = torch.randn(2, 8, 64)
    out = branch(x)
    assert out.shape == (2, 8, 64)
