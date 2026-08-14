"""Concrete Physics-Informed Loss Functions for Radar Perception.

Provides:
- `NonNegativeSignalEnergyLoss`: Penalizes unphysical negative signal power / energy.
- `TemporalSmoothnessLoss`: Penalizes non-physical sudden jumps / acceleration in temporal trajectories.
- `BoundedReflectionLoss`: Constrains radar cross section / reflection intensity within physical bounds [0, I_max].
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_08_pinn_rl.interfaces import PhysicsConstraint


class NonNegativeSignalEnergyLoss(PhysicsConstraint):
    """Penalizes negative signal energy or negative power predictions.

    Physical rationale:
    Photonic radar return power / signal energy is strictly non-negative: E >= 0.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        latent: torch.Tensor,
        prediction: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute non-negative energy penalty."""
        # If prediction dict contains detection / continuous energy, check both
        loss = torch.tensor(0.0, device=latent.device, dtype=latent.dtype)

        # 1. Check latent-derived energy proxy (e.g. latent norm / energy)
        # Any negative component if interpreted as energy
        if isinstance(prediction, dict):
            for k in ["energy", "detection", "anomaly"]:
                if k in prediction and prediction[k] is not None:
                    pred = prediction[k]
                    # Penalize negative values
                    neg_penalty = F.relu(-pred)
                    loss = loss + torch.mean(neg_penalty ** 2)
        elif prediction is not None:
            neg_penalty = F.relu(-prediction)
            loss = loss + torch.mean(neg_penalty ** 2)

        return self.weight * loss


class TemporalSmoothnessLoss(PhysicsConstraint):
    """Enforces kinematic continuity & bounded temporal acceleration.

    Physical rationale:
    Physical radar targets have bounded acceleration; consecutive temporal latent states
    z_t should satisfy ||z_t - 2*z_{t-1} + z_{t-2}|| <= a_max.
    """

    def __init__(self, weight: float = 1.0, max_acceleration: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self.max_acceleration = max_acceleration

    def forward(
        self,
        latent: torch.Tensor,
        prediction: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute temporal smoothness penalty over sequence latent [B, T, H]."""
        if latent.ndim < 3 or latent.shape[1] < 3:
            # If no temporal dimension or sequence length < 3, penalty is zero
            return torch.tensor(0.0, device=latent.device, dtype=latent.dtype)

        # Compute second temporal difference: d2_z = z_{t+1} - 2*z_t + z_{t-1}
        # shape [B, T-2, H]
        d2_z = latent[:, 2:, :] - 2.0 * latent[:, 1:-1, :] + latent[:, :-2, :]
        accel_norm = torch.norm(d2_z, dim=-1)  # [B, T-2]

        # Penalize acceleration exceeding threshold
        excess_accel = F.relu(accel_norm - self.max_acceleration)
        loss = torch.mean(excess_accel ** 2)

        return self.weight * loss


class BoundedReflectionLoss(PhysicsConstraint):
    """Constrains radar reflection intensity / RCS within physically realistic bounds [0, max_intensity].

    Physical rationale:
    Radar Cross Section (RCS) and optical reflection cannot exceed maximum physical bounds.
    """

    def __init__(self, max_intensity: float = 10.0, weight: float = 1.0) -> None:
        super().__init__()
        self.max_intensity = max_intensity
        self.weight = weight

    def forward(
        self,
        latent: torch.Tensor,
        prediction: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute bounded reflection penalty."""
        loss = torch.tensor(0.0, device=latent.device, dtype=latent.dtype)

        # Upper bound check on latent magnitudes
        latent_mag = torch.abs(latent)
        excess_upper = F.relu(latent_mag - self.max_intensity)
        loss = loss + torch.mean(excess_upper ** 2)

        # If prediction has continuous outputs, check bounds
        if isinstance(prediction, dict):
            if "anomaly" in prediction and prediction["anomaly"] is not None:
                ano = prediction["anomaly"]
                excess_ano = F.relu(ano - self.max_intensity)
                loss = loss + torch.mean(excess_ano ** 2)

        return self.weight * loss
