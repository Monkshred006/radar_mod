"""Supervised MLP Policy for PhotonShield V3 Adaptive Compute Scheduling.

Compact 3-layer MLP policy (9 -> 32 -> 16 -> 4) that predicts action probabilities
P(N in {5, 10, 20, 50}) given normalized 9D state vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_07_adaptive_compute.action_space import ACTIONS, IDX_TO_ACTION, ACTION_TO_IDX


class SupervisedDiffusionScheduler(nn.Module):
    """Supervised neural scheduler predicting diffusion step budget from state."""

    def __init__(
        self,
        state_dim: int = 9,
        hidden_dim1: int = 32,
        hidden_dim2: int = 16,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized action logits for batch of states.

        Args:
            state: Float tensor of shape `[B, 9]`.

        Returns:
            Action logits `[B, 4]`.
        """
        return self.net(state)

    def predict_action(
        self,
        state: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[int, torch.Tensor]:
        """Predict optimal diffusion step count for single sample or batch.

        Args:
            state: Float tensor of shape `[1, 9]` or `[9]`.
            deterministic: If True, takes argmax; otherwise samples from softmax.

        Returns:
            Tuple of (selected_action_int, action_probs_tensor).
        """
        self.eval()
        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action_idx = int(torch.argmax(probs, dim=-1)[0].item())
            else:
                action_idx = int(torch.multinomial(probs, num_samples=1)[0].item())

            action_steps = IDX_TO_ACTION[action_idx]

        return action_steps, probs[0]

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)

    def load(self, path: Union[str, Path], device: torch.device) -> None:
        """Load model checkpoint."""
        self.load_state_dict(torch.load(path, map_location=device))
