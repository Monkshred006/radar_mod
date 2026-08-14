"""Composite Physics Constraints Manager for PINN Training.

Combines individual physical laws (Energy conservation, Kinematic continuity, Bounded reflection)
into a unified constraint layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn

from module_08_pinn_rl.interfaces import PhysicsConstraint
from module_08_pinn_rl.physics_losses import (
    NonNegativeSignalEnergyLoss,
    TemporalSmoothnessLoss,
    BoundedReflectionLoss,
)


class CompositePhysicsConstraint(PhysicsConstraint):
    """Composite constraint layer aggregating multiple physics loss terms."""

    def __init__(
        self,
        constraints: Optional[List[PhysicsConstraint]] = None,
        enabled: bool = False,
        lambda_physics: float = 0.1,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.lambda_physics = lambda_physics

        if constraints is None:
            # Default physics constraints suite
            self.constraints = nn.ModuleList([
                NonNegativeSignalEnergyLoss(weight=1.0),
                TemporalSmoothnessLoss(weight=1.0, max_acceleration=2.0),
                BoundedReflectionLoss(weight=0.5, max_intensity=10.0),
            ])
        else:
            self.constraints = nn.ModuleList(constraints)

    def forward(
        self,
        latent: torch.Tensor,
        prediction: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute aggregate weighted physics loss.

        Args:
            latent: Latent tensor `[B, T, H]` or `[B, H]`.
            prediction: Optional prediction tensor or dictionary.
            metadata: Optional metadata dict.

        Returns:
            Total scalar physics loss.
        """
        if not self.enabled:
            return torch.zeros(1, device=latent.device, dtype=latent.dtype)

        total_loss = torch.tensor(0.0, device=latent.device, dtype=latent.dtype)
        for constraint in self.constraints:
            term = constraint(latent, prediction=prediction, metadata=metadata)
            total_loss = total_loss + term

        return self.lambda_physics * total_loss
