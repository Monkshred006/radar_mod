"""Critical Shape and Variable Sequence Length Tests for Module 4."""

import torch
import pytest
from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import ClassificationHead, RegressionHead


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("T", [8, 16, 32, 64])
@pytest.mark.parametrize("D_feat", [48])
@pytest.mark.parametrize("S", [5])
def test_end_to_end_shapes(B, T, D_feat, S):
    d_model = 64
    num_classes = 3

    config = MambaHybridConfig(
        d_model=d_model,
        sensor_feature_dim=D_feat,
        num_sensor_groups=S,
        num_layers=2,
        backend="fallback",
    )

    engine = PhotonMambaHybrid(config)
    head = ClassificationHead(d_model, TaskHeadConfig(num_classes=num_classes))

    # Simulate Module 3 tokens [B, T, S, D_features]
    tokens = torch.randn(B, T, S, D_feat)
    token_mask = torch.ones(B, T, S, D_feat, dtype=torch.bool)
    m3_out = {"tokens": tokens, "token_mask": token_mask}

    # Step 1: Engine forward pass
    engine_out = engine(m3_out)

    seq_out = engine_out["sequence_output"]
    pooled_out = engine_out["pooled_output"]

    assert seq_out.shape == (B, T, d_model), f"Expected ({B}, {T}, {d_model}), got {seq_out.shape}"
    assert pooled_out.shape == (B, d_model), f"Expected ({B}, {d_model}), got {pooled_out.shape}"

    # Step 2: Head forward pass
    logits = head(pooled_out)
    assert logits.shape == (B, num_classes), f"Expected ({B}, {num_classes}), got {logits.shape}"


def test_gradient_flow_tokens():
    config = MambaHybridConfig(d_model=32, num_layers=1, backend="fallback")
    engine = PhotonMambaHybrid(config)
    head = RegressionHead(32, TaskHeadConfig(num_regression_outputs=1))

    tokens = torch.randn(2, 10, 5, 48, requires_grad=True)
    m3_out = {"tokens": tokens}

    out = engine(m3_out)
    pred = head(out["pooled_output"])

    loss = pred.sum()
    loss.backward()

    # Verify gradients flow back to input tokens
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()

    # Verify parameters used in token path receive finite gradients
    for name, param in engine.named_parameters():
        if "flat_proj" not in name and param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf gradient"


def test_gradient_flow_flat_features():
    config = MambaHybridConfig(d_model=32, num_layers=1, backend="fallback")
    engine = PhotonMambaHybrid(config)
    head = RegressionHead(32, TaskHeadConfig(num_regression_outputs=1))

    features = torch.randn(2, 10, 101, requires_grad=True)
    m3_out = {"features": features}

    out = engine(m3_out)
    pred = head(out["pooled_output"])

    loss = pred.sum()
    loss.backward()

    # Verify gradients flow back to input features and flat_proj
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert engine.input_projection.flat_proj.weight.grad is not None
