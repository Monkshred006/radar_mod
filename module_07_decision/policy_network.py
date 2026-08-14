"""Sensor Control Policy Network (Phase V2 Preparation).

Actor-Critic policy architecture mapping RL state representations to discrete
sensor adaptation control decisions.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# Discrete Action Catalogs
GAIN_OPTIONS = [-12.0, -6.0, 0.0, 6.0, 12.0]  # dB
PULSE_WIDTH_OPTIONS = [2.0, 5.0, 10.0, 20.0]    # microseconds
SAMPLING_RATE_OPTIONS = [10.0, 20.0, 40.0, 80.0] # MHz
FRAME_AVG_OPTIONS = [1, 2, 4, 8]                 # frames


class SensorPolicyNetwork(nn.Module):
    """Multi-Head Actor-Critic Policy Network for Radar Control."""

    def __init__(
        self,
        state_dim: int = 19,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Trunk Network
        layers = [nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)

        # Actor Action Heads
        self.gain_head = nn.Linear(hidden_dim, len(GAIN_OPTIONS))
        self.pulse_width_head = nn.Linear(hidden_dim, len(PULSE_WIDTH_OPTIONS))
        self.sampling_rate_head = nn.Linear(hidden_dim, len(SAMPLING_RATE_OPTIONS))
        self.frame_avg_head = nn.Linear(hidden_dim, len(FRAME_AVG_OPTIONS))

        # Critic Value Head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            state: RL state tensor `[B, state_dim]`.

        Returns:
            Tuple of:
                - action_logits: Dict mapping action name -> logits `[B, num_options]`
                - state_value: `[B, 1]`
        """
        features = self.trunk(state)

        logits = {
            "gain": self.gain_head(features),
            "pulse_width": self.pulse_width_head(features),
            "sampling_rate": self.sampling_rate_head(features),
            "frame_avg": self.frame_avg_head(features),
        }
        value = self.value_head(features)
        return logits, value

    def sample_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """Sample actions from categorical policy distributions.

        Args:
            state: `[B, state_dim]`.
            deterministic: If True, select argmax actions.

        Returns:
            Tuple of:
                - actions: Dict mapping action name -> action index tensor `[B]`
                - log_probs: Dict mapping action name -> log prob tensor `[B]`
                - value: Value estimate `[B, 1]`
        """
        logits, value = self.forward(state)
        actions = {}
        log_probs = {}

        for key, logit in logits.items():
            dist = Categorical(logits=logit)
            if deterministic:
                act = torch.argmax(logit, dim=-1)
            else:
                act = dist.sample()
            actions[key] = act
            log_probs[key] = dist.log_prob(act)

        return actions, log_probs, value
