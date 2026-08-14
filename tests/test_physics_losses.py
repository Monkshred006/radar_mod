"""Unit tests for PINN physics loss constraints."""

import pytest
import torch
import torch.nn as nn

from module_08_pinn_rl.interfaces import PhysicsConstraint
from module_08_pinn_rl.physics_losses import (
    NonNegativeSignalEnergyLoss,
    TemporalSmoothnessLoss,
    BoundedReflectionLoss,
)
from module_08_pinn_rl.pinn_constraints import CompositePhysicsConstraint


class TestPhysicsLosses:
    """Test suite for physics loss functions."""

    def test_non_negative_energy_loss(self):
        loss_fn = NonNegativeSignalEnergyLoss(weight=1.0)
        latent = torch.randn(4, 16, 64)

        # Non-negative prediction should have zero loss
        valid_pred = {"energy": torch.tensor([[1.0], [2.0], [0.5], [3.0]])}
        loss_zero = loss_fn(latent, prediction=valid_pred)
        assert loss_zero.item() == 0.0

        # Negative prediction should trigger penalty
        neg_pred = {"energy": torch.tensor([[-1.0], [2.0], [-0.5], [3.0]])}
        loss_pos = loss_fn(latent, prediction=neg_pred)
        assert loss_pos.item() > 0.0

    def test_temporal_smoothness_loss(self):
        loss_fn = TemporalSmoothnessLoss(weight=1.0, max_acceleration=0.5)

        # Smooth linear trajectory: acceleration is exactly 0
        t = torch.linspace(0, 10, 16).view(1, 16, 1).repeat(2, 1, 64)
        loss_smooth = loss_fn(t)
        assert loss_smooth.item() == 0.0

        # Random jerky trajectory
        jerky = torch.randn(2, 16, 64) * 10.0
        loss_jerky = loss_fn(jerky)
        assert loss_jerky.item() > 0.0

    def test_bounded_reflection_loss(self):
        loss_fn = BoundedReflectionLoss(max_intensity=5.0, weight=1.0)

        # Values within [0, 5.0]
        bounded_latent = torch.ones(2, 16, 64) * 2.0
        loss_zero = loss_fn(bounded_latent)
        assert loss_zero.item() == 0.0

        # Values exceeding 5.0
        excess_latent = torch.ones(2, 16, 64) * 10.0
        loss_excess = loss_fn(excess_latent)
        assert loss_excess.item() > 0.0

    def test_composite_physics_constraint(self):
        latent = torch.randn(2, 16, 64)

        # Disabled
        composite_off = CompositePhysicsConstraint(enabled=False)
        assert composite_off(latent).item() == 0.0

        # Enabled
        composite_on = CompositePhysicsConstraint(enabled=True, lambda_physics=0.1)
        loss = composite_on(latent)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
