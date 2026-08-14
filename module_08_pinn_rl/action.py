"""Action specification and encoding for Module 8 PINN + RL.

Action semantics
----------------
The discrete action defaults below are DEVELOPMENT / SYNTHETIC PLACEHOLDERS:
    0 = maintain
    1 = inspect
    2 = alert
    3 = reposition

These do NOT represent real PhotonShield control actions. The actual
physical action space must be defined when the real application task
environment is specified.

For the synthetic kinematic environment, actions map to acceleration
forces: {0: -1, 1: 0, 2: +1}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch


DISCRETE_ACTION_LABELS = {
    0: "maintain",      # PLACEHOLDER — real semantics TBD
    1: "inspect",       # PLACEHOLDER — real semantics TBD
    2: "alert",         # PLACEHOLDER — real semantics TBD
    3: "reposition",    # PLACEHOLDER — real semantics TBD
}

SYNTHETIC_KINEMATIC_ACTION_MAP = {
    0: -1.0,   # decelerate
    1:  0.0,   # maintain velocity
    2:  1.0,   # accelerate
}


@dataclass
class ActionSpec:
    """Specifies the action space for an RL experiment.

    Attributes
    ----------
    action_type : "discrete" or "continuous"
        Determines how actions are encoded and interpreted.
    action_dim : int
        Discrete: number of distinct actions.
        Continuous: dimensionality of the action vector.
    discrete_names : List[str], optional
        Human-readable labels for discrete actions (dev placeholders).
    continuous_low : List[float], optional
        Lower bounds for continuous action dimensions.
    continuous_high : List[float], optional
        Upper bounds for continuous action dimensions.
    """

    action_type: str = "discrete"
    action_dim: int = 4
    discrete_names: Optional[List[str]] = None
    continuous_low: Optional[List[float]] = None
    continuous_high: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.action_type not in ("discrete", "continuous"):
            raise ValueError("action_type must be 'discrete' or 'continuous'.")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0.")

    @classmethod
    def discrete(cls, n_actions: int, names: Optional[List[str]] = None) -> "ActionSpec":
        return cls(action_type="discrete", action_dim=n_actions, discrete_names=names)

    @classmethod
    def continuous(
        cls,
        dim: int,
        low: Optional[List[float]] = None,
        high: Optional[List[float]] = None,
    ) -> "ActionSpec":
        return cls(
            action_type="continuous",
            action_dim=dim,
            continuous_low=low,
            continuous_high=high,
        )


class ActionEncoder:
    """Encodes and decodes actions for use by the PINN dynamics model.

    Discrete actions are one-hot encoded before being fed into the
    dynamics MLP so that the input is a fixed-size float tensor regardless
    of action_type.
    """

    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec

    @property
    def encoded_dim(self) -> int:
        """Dimensionality of the encoded action vector."""
        if self.spec.action_type == "discrete":
            return self.spec.action_dim   # one-hot
        return self.spec.action_dim       # raw continuous

    def encode(self, action: Any) -> torch.Tensor:
        """Encode a raw action into a float tensor.

        Parameters
        ----------
        action : int (discrete) or Tensor / array (continuous)

        Returns
        -------
        Tensor[encoded_dim]
        """
        if self.spec.action_type == "discrete":
            idx = int(action)
            if not 0 <= idx < self.spec.action_dim:
                raise ValueError(
                    f"Discrete action {idx} out of range [0, {self.spec.action_dim})."
                )
            one_hot = torch.zeros(self.spec.action_dim, dtype=torch.float32)
            one_hot[idx] = 1.0
            return one_hot
        else:
            a = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
            if a.shape[0] != self.spec.action_dim:
                raise ValueError(
                    f"Continuous action has dim {a.shape[0]}, "
                    f"expected {self.spec.action_dim}."
                )
            return a

    def decode(self, encoded: torch.Tensor) -> Any:
        """Decode an encoded action tensor back to raw form.

        Discrete: returns int index.
        Continuous: returns the raw tensor.
        """
        if self.spec.action_type == "discrete":
            return int(encoded.argmax().item())
        return encoded.clone()
