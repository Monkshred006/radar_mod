"""Physics-Informed Dynamics Model for Module 8.

Implements  f_θ(s_t, a_t) → ŝ_{t+1}  as a configurable MLP.

PINN role in Module 8
---------------------
The PINN is used as a LEARNED PHYSICS-INFORMED DYNAMICS MODEL.
It is trained to predict next states while minimising a physics residual:
    L_total = L_data + λ_phys · L_physics

This is distinct from:
- A standalone PDE solver
- A sensor measurement classifier
- A reward estimator

Experiment roles
----------------
Exp B (data-only): λ_phys = 0 → PINN learns from data only
Exp C (RL+PINN):   λ_phys > 0 → PINN is physics-informed

Action encoding
---------------
Discrete actions are one-hot encoded to produce fixed-size float input.
The encoder is created internally from the DynamicsConfig.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from module_08_pinn_rl.action import ActionEncoder, ActionSpec
from module_08_pinn_rl.config import DynamicsConfig


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: List[int],
    activation: str,
) -> nn.Sequential:
    """Build a configurable MLP."""
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    if activation not in act_map:
        raise ValueError(f"Unknown activation: '{activation}'.")
    Act = act_map[activation]

    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(Act())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PhysicsInformedDynamicsModel(nn.Module):
    """Learned dynamics model  f_θ(s_t, a_t) → ŝ_{t+1}.

    Architecture: configurable MLP.
    Input:  concatenation of [state, encoded_action]
    Output: predicted next state of the same dimension as the input state.

    Parameters
    ----------
    config : DynamicsConfig
        Specifies state_dim, action_type, action_dim, hidden_dims,
        activation.
    """

    def __init__(self, config: DynamicsConfig) -> None:
        super().__init__()
        self.config = config

        spec = ActionSpec(action_type=config.action_type, action_dim=config.action_dim)
        self.action_encoder = ActionEncoder(spec)

        in_dim = config.state_dim + self.action_encoder.encoded_dim
        self.net = _build_mlp(
            in_dim=in_dim,
            out_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state.

        Parameters
        ----------
        state : Tensor[B, state_dim] or Tensor[state_dim]
            Current state vector.
        action : Tensor[B, encoded_dim] or Tensor[encoded_dim]
            Encoded action (one-hot for discrete, raw for continuous).
            If a raw integer is passed it will be encoded automatically.

        Returns
        -------
        Tensor[B, state_dim]
            Predicted next state ŝ_{t+1}.
        """
        state = state.float()
        action = action.float()

        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    def predict(self, state: np.ndarray, action: Any) -> np.ndarray:
        """Numpy interface for environment integration.

        Parameters
        ----------
        state : ndarray[state_dim]
        action : int (discrete) or ndarray (continuous)

        Returns
        -------
        ndarray[state_dim]
        """
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32))
            a = self.action_encoder.encode(action)
            pred = self.forward(s, a)
            return pred.squeeze(0).numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
