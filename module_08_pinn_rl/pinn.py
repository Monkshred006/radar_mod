"""PINN loss for Module 8.

Computes:
    L_total = L_data + λ_phys · L_physics

L_data   = prediction error (MSE / L1 / SmoothL1 — configurable)
L_physics = mean(|residual|²)

Key design rules
----------------
* PINN loss backpropagates to the DYNAMICS MODEL parameters only.
  It does NOT automatically backpropagate through the RL policy.
* λ_phys = 0 → data-only baseline (Experiment B). Not the same as RL-only.
* λ_phys > 0 → physics-informed (Experiment C).
* L_physics and RL reward R_t are STRICTLY SEPARATE quantities.

The `forward()` method returns a dict so all sub-losses are available
for logging without re-computation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from module_08_pinn_rl.config import DynamicsConfig, PhysicsConfig
from module_08_pinn_rl.physics import PhysicsModel, build_physics_model


def _get_data_loss_fn(name: str):
    """Return the data loss function by name."""
    if name == "mse":
        return F.mse_loss
    elif name == "l1":
        return F.l1_loss
    elif name == "smooth_l1":
        return F.smooth_l1_loss
    else:
        raise ValueError(f"Unknown data_loss: '{name}'. Choose mse, l1, smooth_l1.")


class PINNLoss(nn.Module):
    """Physics-Informed Neural Network loss for dynamics model training.

    L_total = L_data + λ_phys · L_physics

    Parameters
    ----------
    dynamics_config : DynamicsConfig
        Provides data_loss type.
    physics_config : PhysicsConfig
        Provides lambda_physics and physics_model selection.
    physics_model : PhysicsModel, optional
        Pre-built physics model. If None, builds from physics_config.
    """

    def __init__(
        self,
        dynamics_config: DynamicsConfig,
        physics_config: PhysicsConfig,
        physics_model: Optional[PhysicsModel] = None,
    ) -> None:
        super().__init__()
        self.lambda_physics = physics_config.lambda_physics
        self.data_loss_fn = _get_data_loss_fn(dynamics_config.data_loss)
        self.physics_model = (
            physics_model
            if physics_model is not None
            else build_physics_model(physics_config)
        )

    def forward(
        self,
        predicted_next_state: torch.Tensor,
        observed_next_state: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        **physics_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute PINN loss.

        Parameters
        ----------
        predicted_next_state : Tensor[B, state_dim]
            ŝ_{t+1} from the dynamics model.
        observed_next_state : Tensor[B, state_dim]
            Ground-truth s_{t+1} from the environment or dataset.
        state : Tensor[B, state_dim]
            Current state s_t.
        action : Tensor[B, encoded_dim]
            Encoded action a_t.
        **physics_kwargs
            Additional keyword arguments forwarded to physics_model.residual()
            (e.g. u_fn, x, t for wave-convection model).

        Returns
        -------
        dict with keys:
            "loss"          — L_total (scalar, differentiable)
            "data_loss"     — L_data (scalar)
            "physics_loss"  — L_physics (scalar, always computed for logging)
        """
        # Data loss
        l_data = self.data_loss_fn(predicted_next_state, observed_next_state)

        # Physics loss (computed even when lambda=0 for logging)
        residual = self.physics_model.residual(
            state=state,
            action=action,
            next_state_pred=predicted_next_state,
            **physics_kwargs,
        )
        l_physics = residual.float().pow(2).mean()

        l_total = l_data + self.lambda_physics * l_physics

        return {
            "loss": l_total,
            "data_loss": l_data,
            "physics_loss": l_physics,
        }
