"""RL Policy network for Module 8.

MLPPolicy — Actor-Critic architecture, configurable for discrete or continuous
action spaces.

The policy is NOT coupled to the PINN. PINN physics loss does NOT
backpropagate through the RL policy by default. The policy receives
only RL gradients from the PPO objective.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from module_08_pinn_rl.config import RLConfig


def _build_mlp(in_dim: int, out_dim: int, hidden_dims: List[int], activation: str) -> nn.Sequential:
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh}
    if activation not in act_map:
        raise ValueError(f"Unknown activation '{activation}'.")
    Act = act_map[activation]
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(Act())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MLPPolicy(nn.Module):
    """Actor-Critic MLP policy.

    For discrete actions: actor outputs logits → Categorical distribution.
    For continuous actions: actor outputs mean; log_std is a learned parameter
    → Normal distribution.

    Parameters
    ----------
    state_dim : int
    config : RLConfig
        Provides action_type, action_dim, hidden_dims, activation.
    """

    def __init__(self, state_dim: int, config: RLConfig) -> None:
        super().__init__()
        self.config = config
        self.action_type = config.action_type
        self.action_dim = config.action_dim

        # Shared feature extractor (optional — currently separate nets)
        self.actor = _build_mlp(
            in_dim=state_dim,
            out_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        )
        self.critic = _build_mlp(
            in_dim=state_dim,
            out_dim=1,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        )

        if config.action_type == "continuous":
            self.log_std = nn.Parameter(
                torch.zeros(config.action_dim)
            )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return actor output and value estimate.

        Returns
        -------
        (actor_out, value) where actor_out is logits (discrete) or mean (continuous).
        """
        actor_out = self.actor(state)
        value = self.critic(state).squeeze(-1)
        return actor_out, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate an action and compute log-prob, entropy, value.

        Parameters
        ----------
        state : Tensor[B, state_dim] or Tensor[state_dim]
        action : Tensor, optional
            If provided, evaluates log-prob for the given action (training).
            If None, samples a new action (rollout).

        Returns
        -------
        (action, log_prob, entropy, value)
        """
        actor_out, value = self.forward(state)

        if self.action_type == "discrete":
            dist = Categorical(logits=actor_out)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            std = self.log_std.exp().expand_as(actor_out)
            dist = Normal(actor_out, std)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value

    def act(self, state: torch.Tensor) -> int:
        """Deterministic greedy action for evaluation."""
        with torch.no_grad():
            actor_out, _ = self.forward(state.unsqueeze(0) if state.ndim == 1 else state)
            if self.action_type == "discrete":
                return int(actor_out.argmax(dim=-1).item())
            else:
                return actor_out.squeeze(0).numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
