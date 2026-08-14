"""Unit Tests for Gap-Aware Physics Weighting (PhotonShield V2.4).

Tests the gap-aware temporal weighting mechanism in RadarPhysicsLoss.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch
from module_06_physics.physics_losses import RadarPhysicsLoss
from module_06_physics.latent_physics_head import LatentPhysicsHead

B = 2
T = 16
LATENT_DIM = 64


def _make_loss_fn(gap_alpha: float = 0.5) -> RadarPhysicsLoss:
    """Create a RadarPhysicsLoss with a fresh LatentPhysicsHead."""
    torch.manual_seed(42)
    physics_head = LatentPhysicsHead(latent_dim=LATENT_DIM, hidden_dim=32)
    return RadarPhysicsLoss(gap_alpha=gap_alpha, physics_head=physics_head)


# ============================================================
# Test 1: No dropout => all gap weights = 1.0
# ============================================================
def test_gap_weights_no_dropout():
    torch.manual_seed(42)
    loss_fn = _make_loss_fn(gap_alpha=0.5)
    mask = torch.ones(B, T, 1)

    weights = loss_fn.compute_gap_weights(mask)

    assert weights.shape == (B, T - 1)
    assert torch.all(weights == 1.0)


# ============================================================
# Test 2: Single missing frame => affected transitions < 1.0
# ============================================================
def test_gap_weights_single_missing():
    torch.manual_seed(42)
    loss_fn = _make_loss_fn(gap_alpha=0.5)
    mask = torch.ones(B, T, 1)
    mask[:, 2, :] = 0.0  # frame 2 is missing

    weights = loss_fn.compute_gap_weights(mask)

    assert weights.shape == (B, T - 1)
    # Transitions 1->2 and 2->3 touch the gap
    assert torch.all(weights[:, 1] < 1.0)
    assert torch.all(weights[:, 2] < 1.0)
    # Transitions that don't touch the gap
    assert torch.all(weights[:, 0] == 1.0)
    assert torch.all(weights[:, 3:] == 1.0)


# ============================================================
# Test 3: Increasing gap length => monotonically decreasing weights
# ============================================================
def test_gap_weights_monotonic_decrease():
    torch.manual_seed(42)
    loss_fn = _make_loss_fn(gap_alpha=0.5)

    weights_at_gap = []

    for gap_len in [1, 2, 3, 5]:
        mask = torch.ones(B, T, 1)
        for i in range(2, 2 + gap_len):
            mask[:, i, :] = 0.0

        w = loss_fn.compute_gap_weights(mask)
        # Pick a transition inside the gap
        weights_at_gap.append(w[0, 2].item())

    assert weights_at_gap[0] > weights_at_gap[1]
    assert weights_at_gap[1] > weights_at_gap[2]
    assert weights_at_gap[2] > weights_at_gap[3]


# ============================================================
# Test 4: Very long gap => weights remain positive and finite
# ============================================================
def test_gap_weights_long_gap_positive_finite():
    torch.manual_seed(42)
    loss_fn = _make_loss_fn(gap_alpha=0.5)
    mask = torch.zeros(B, T, 1)
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0

    weights = loss_fn.compute_gap_weights(mask)

    assert torch.all(weights > 0.0)
    assert torch.all(torch.isfinite(weights))


# ============================================================
# Test 5: Observed frame preservation (backward compatibility)
# ============================================================
def test_gap_weights_observed_frame_preservation():
    """When all frames are observed, gap-aware loss equals non-gap-aware loss."""
    torch.manual_seed(42)
    loss_fn = _make_loss_fn(gap_alpha=0.5)

    z = torch.randn(B, T, LATENT_DIM)
    mask = torch.ones(B, T, 1)

    # Compute observables
    obs = loss_fn.physics_head(z)
    r_hat = obs["range"]
    v_hat = obs["velocity"]

    # Without gap weights (original)
    loss_uniform, _ = loss_fn.compute_kinematic_loss(r_hat, v_hat, gap_weights=None)

    # With gap weights from all-observed mask
    weights = loss_fn.compute_gap_weights(mask)
    loss_gap, _ = loss_fn.compute_kinematic_loss(r_hat, v_hat, gap_weights=weights)

    assert torch.allclose(loss_uniform, loss_gap, rtol=1e-4, atol=1e-6)


# ============================================================
# Test 6: No NaN/Inf in edge cases
# ============================================================
def test_gap_weights_no_nan_inf():
    torch.manual_seed(42)
    loss_fn = _make_loss_fn(gap_alpha=0.5)

    z = torch.randn(B, T, LATENT_DIM)

    edge_masks = [
        # All missing except frame 0
        torch.cat([torch.ones(B, 1, 1), torch.zeros(B, T - 1, 1)], dim=1),
        # All observed
        torch.ones(B, T, 1),
        # Alternating observed/missing
        torch.tensor([[[1.0], [0.0]] * (T // 2)]).expand(B, -1, -1).clone(),
    ]

    for mask in edge_masks:
        weights = loss_fn.compute_gap_weights(mask)
        assert not torch.isnan(weights).any(), "NaN in gap weights"
        assert not torch.isinf(weights).any(), "Inf in gap weights"

        total_loss, components = loss_fn(z, x_clean=None, mask=mask)
        assert not torch.isnan(total_loss).any(), "NaN in total loss"
        assert not torch.isinf(total_loss).any(), "Inf in total loss"


# ============================================================
# Test 7: Gap-aware weighting down-weights long-gap transitions
# ============================================================
def test_gap_aware_weighting_reduces_gap_influence():
    """Gap-aware weighting produces lower weights on gap transitions and differs from uniform."""
    torch.manual_seed(42)
    loss_fn = _make_loss_fn(gap_alpha=0.5)

    z = torch.randn(B, T, LATENT_DIM)
    obs = loss_fn.physics_head(z)
    r_hat = obs["range"]
    v_hat = obs["velocity"]

    mask = torch.ones(B, T, 1)
    mask[:, 5:12, :] = 0.0  # 7-frame gap

    weights = loss_fn.compute_gap_weights(mask)

    # Transitions inside the gap (indices 4-11) should have w < 1
    gap_transition_weights = weights[:, 4:12]
    assert torch.all(gap_transition_weights < 1.0), "Gap transitions should have weight < 1"

    # Non-gap transitions should have weight = 1
    assert torch.all(weights[:, :4] == 1.0), "Non-gap transitions should have weight = 1"
    assert torch.all(weights[:, 12:] == 1.0), "Non-gap transitions should have weight = 1"

    # Gap-aware loss should differ from uniform (weighting changes the result)
    loss_gap, _ = loss_fn.compute_kinematic_loss(r_hat, v_hat, gap_weights=weights)
    loss_fixed, _ = loss_fn.compute_kinematic_loss(r_hat, v_hat, gap_weights=None)
    assert not torch.allclose(loss_gap, loss_fixed, atol=1e-6), "Gap-aware and fixed loss should differ"


# ============================================================
# Test 8: Gradient flow through gap-weighted loss
# ============================================================
def test_gap_aware_gradient_flow():
    torch.manual_seed(42)
    loss_fn = _make_loss_fn(gap_alpha=0.5)

    z = torch.randn(B, T, LATENT_DIM, requires_grad=True)

    mask = torch.ones(B, T, 1)
    mask[:, 3:7, :] = 0.0

    total_loss, components = loss_fn(z, x_clean=None, mask=mask)
    total_loss.backward()

    assert z.grad is not None
    assert not torch.isnan(z.grad).any()
    assert not torch.isinf(z.grad).any()


# ============================================================
# Test 9: gap_alpha=0 => uniform weights, identical to no-mask
# ============================================================
def test_gap_alpha_zero_equals_uniform():
    torch.manual_seed(42)
    loss_fn = _make_loss_fn(gap_alpha=0.0)

    z = torch.randn(B, T, LATENT_DIM)

    mask = torch.ones(B, T, 1)
    mask[:, 4:10, :] = 0.0

    # With alpha=0, gap_alpha > 0 is False, so forward() won't compute gap weights
    # Test compute_gap_weights directly
    weights = loss_fn.compute_gap_weights(mask)
    assert torch.allclose(weights, torch.ones_like(weights))

    # Loss with mask (alpha=0 means no gap-aware behavior in forward)
    loss_with_mask, comp_mask = loss_fn(z, x_clean=None, mask=mask)
    loss_no_mask, comp_no_mask = loss_fn(z, x_clean=None, mask=None)

    assert torch.allclose(comp_mask["loss_kinematic"], comp_no_mask["loss_kinematic"], rtol=1e-4, atol=1e-6)
    assert torch.allclose(comp_mask["loss_acceleration"], comp_no_mask["loss_acceleration"], rtol=1e-4, atol=1e-6)
