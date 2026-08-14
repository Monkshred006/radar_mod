"""PINN Physics Constraint Interfaces for PhotonShield AI (Phase V2 Preparation).

Defines base abstract class for physics constraints applied to latent representations
and predictions during neural network training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class PhysicsConstraint(nn.Module, ABC):
    """Abstract base class for PINN physics loss constraints.

    All concrete implementations must override `forward()`.
    """

    @abstractmethod
    def forward(
        self,
        latent: torch.Tensor,
        prediction: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute physics residual / penalty loss term.

        Args:
            latent: Latent state tensor `[B, T, H]` or `[B, H]`.
            prediction: Optional prediction tensor or dictionary of predictions.
            metadata: Optional dictionary with physical parameters (dt, c, SNR, RCS, etc.).

        Returns:
            Scalar penalty tensor (>= 0).
        """
        pass
