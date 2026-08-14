"""Transition and Episode dataclasses for Module 8 PINN + RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Transition:
    """A single environment transition (s_t, a_t, r_t, s_{t+1}, done).

    All fields are stored as numpy arrays / scalars for framework
    independence; conversion to tensors happens in the replay buffer.
    """

    state: np.ndarray          # RL state at time t
    action: Any                # Action taken (int for discrete, ndarray for continuous)
    reward: float              # Scalar reward r_t
    next_state: np.ndarray     # RL state at time t+1
    done: bool                 # Episode termination flag
    log_prob: float = 0.0      # Log-probability of the action (for on-policy algorithms)
    value: float = 0.0         # Value estimate V(s_t) (for Actor-Critic)
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A complete RL episode consisting of an ordered sequence of transitions."""

    transitions: List[Transition] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)

    @property
    def length(self) -> int:
        return len(self.transitions)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def is_empty(self) -> bool:
        return len(self.transitions) == 0
