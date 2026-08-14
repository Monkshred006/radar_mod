"""Abstract base classes and data containers for Module 9."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


@dataclass
class SceneInput:
    """Standardized input container consuming upstream outputs from Modules 4, 7, and 8.

    None of the upstream fields are mandatory. All are optional, allowing flexible
    operation from pure sensor representations, decisions, physical states, or direct 3D points.
    """

    # Module 4 Mamba-Hybrid representations
    latent: Optional[Union[torch.Tensor, np.ndarray]] = None  # pooled_output [D_model]
    sequence_latent: Optional[Union[torch.Tensor, np.ndarray]] = None  # [T, D_model]

    # Module 7 Decision outputs
    target_probability: Optional[float] = None
    anomaly_probability: Optional[float] = None
    environmental_assessment: Optional[List[float]] = None
    decision_detected: Optional[bool] = None

    # Module 8 PINN + RL outputs
    physical_state: Optional[Union[torch.Tensor, np.ndarray]] = None  # [x, v, ...]
    predicted_next_state: Optional[Union[torch.Tensor, np.ndarray]] = None
    rl_action: Optional[Any] = None

    # Raw point-cloud pass-through (if pre-computed or synthetic)
    raw_points: Optional[np.ndarray] = None  # [N, 3] or [N, 3+F]

    # Metadata
    timestamp: float = 0.0
    frame_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreeDReconstructor(ABC):
    """Abstract interface for 3D reconstruction backends."""

    @abstractmethod
    def reconstruct(self, scene_input: SceneInput) -> "Scene3D":  # type: ignore[name-defined]
        """Reconstruct a Scene3D from SceneInput representations."""
        pass


class DisplayBackend(ABC):
    """Abstract interface for visual display devices (OLED, software canvas, etc.)."""

    @abstractmethod
    def display_frame(self, frame: np.ndarray) -> bool:
        """Display an RGB image frame [H, W, 3]. Returns True if successful."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Check if display backend is connected/open."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up display resources."""
        pass


class ReconstructionEvaluator(ABC):
    """Abstract evaluation interface for 3D reconstruction."""

    @abstractmethod
    def evaluate(
        self,
        predicted_scene: "Scene3D",  # type: ignore[name-defined]
        ground_truth_scene: Optional["Scene3D"] = None,  # type: ignore[name-defined]
    ) -> Dict[str, Any]:
        """Evaluate reconstruction accuracy.

        If ground_truth_scene is None, returns status='ground_truth_unavailable'
        without fabricating metrics.
        """
        pass
