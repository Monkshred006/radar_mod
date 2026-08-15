"""Action space definition for PhotonShield V3 Adaptive Compute."""

from __future__ import annotations

from typing import List, Dict
import torch

ACTIONS: List[int] = [5, 10, 20, 50]
ACTION_TO_IDX: Dict[int, int] = {5: 0, 10: 1, 20: 2, 50: 3}
IDX_TO_ACTION: Dict[int, int] = {0: 5, 1: 10, 2: 20, 3: 50}
NUM_ACTIONS: int = len(ACTIONS)


def get_action_space() -> List[int]:
    """Return the discrete diffusion steps action space."""
    return list(ACTIONS)


def action_to_index(action: int) -> int:
    """Map diffusion step count to action index (0..3)."""
    if action not in ACTION_TO_IDX:
        raise ValueError(f"Invalid action {action}. Must be one of {ACTIONS}")
    return ACTION_TO_IDX[action]


def index_to_action(idx: int) -> int:
    """Map action index (0..3) to diffusion step count."""
    if idx not in IDX_TO_ACTION:
        raise ValueError(f"Invalid action index {idx}. Must be 0..{NUM_ACTIONS-1}")
    return IDX_TO_ACTION[idx]
