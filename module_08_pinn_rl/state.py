"""RL State construction from Module 4 / Module 7 outputs.

RLStateBuilder assembles the RL state vector from continuous outputs
produced by the upstream modules. All state components are configurable
via RLStateConfig. The state dimension is always derived from configuration
and never hard-coded.

Key design principles
---------------------
* Continuous information is preserved — not thresholded binary decisions.
* Causality: the state at time t depends only on information at or before t.
* Immutability: building a new state at t+1 does not affect any previously
  built state at t (no shared mutable arrays).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

from module_08_pinn_rl.config import RLStateConfig


@dataclass
class RLState:
    """The RL state vector at a single timestep.

    Attributes
    ----------
    vector : Tensor[state_dim]
        Flat concatenated state vector consumed by the RL policy and
        PINN dynamics model.
    components : Dict[str, Tensor]
        Named sub-tensors for inspection, debugging, and logging.
    timestamp : float, optional
        Wall-clock or simulation timestamp; purely informational.
    """

    vector: torch.Tensor
    components: Dict[str, torch.Tensor] = field(default_factory=dict)
    timestamp: Optional[float] = None


class RLStateBuilder:
    """Assembles an RLState from Module 4 / Module 7 outputs.

    Parameters
    ----------
    config : RLStateConfig
        Specifies which components are included and their dimensions.
    """

    def __init__(self, config: RLStateConfig) -> None:
        config.validate()
        self.config = config

    @property
    def state_dim(self) -> int:
        """Derived state dimension — never hard-coded."""
        return self.config.state_dim

    def build(
        self,
        pooled_output: Optional[torch.Tensor] = None,
        target_probability: Optional[float] = None,
        anomaly_probability: Optional[float] = None,
        environment_output: Optional[Union[List[float], torch.Tensor]] = None,
        physical_state: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None,
    ) -> RLState:
        """Build an RLState from available upstream outputs.

        Parameters
        ----------
        pooled_output : Tensor[D_model] or Tensor[1, D_model]
            Module 4 pooled latent representation.
        target_probability : float
            Module 7 continuous target probability (NOT a binary decision).
        anomaly_probability : float
            Module 7 continuous anomaly probability (NOT a binary decision).
        environment_output : List[float] or Tensor[env_dim]
            Module 7 environmental outputs.
        physical_state : Tensor[phys_dim], optional
            Optional physical state variables (application-dependent).
        timestamp : float, optional
            Simulation or wall-clock time; purely informational.

        Returns
        -------
        RLState
            Immutable state object. Building a new state does not modify
            any previously returned RLState.
        """
        parts: List[torch.Tensor] = []
        components: Dict[str, torch.Tensor] = {}

        cfg = self.config

        if cfg.use_mamba_latent:
            if pooled_output is None:
                raise ValueError(
                    "pooled_output is required when use_mamba_latent=True."
                )
            z = pooled_output.detach().reshape(-1).float()
            if z.shape[0] != cfg.mamba_latent_dim:
                raise ValueError(
                    f"pooled_output has dim {z.shape[0]}, "
                    f"expected mamba_latent_dim={cfg.mamba_latent_dim}."
                )
            parts.append(z)
            components["mamba_latent"] = z.clone()

        if cfg.use_target_probability:
            if target_probability is None:
                raise ValueError(
                    "target_probability is required when use_target_probability=True."
                )
            p = torch.tensor([float(target_probability)], dtype=torch.float32)
            parts.append(p)
            components["target_probability"] = p.clone()

        if cfg.use_anomaly_probability:
            if anomaly_probability is None:
                raise ValueError(
                    "anomaly_probability is required when use_anomaly_probability=True."
                )
            p = torch.tensor([float(anomaly_probability)], dtype=torch.float32)
            parts.append(p)
            components["anomaly_probability"] = p.clone()

        if cfg.use_environment:
            if environment_output is None:
                raise ValueError(
                    "environment_output is required when use_environment=True."
                )
            if isinstance(environment_output, torch.Tensor):
                e = environment_output.detach().reshape(-1).float()
            else:
                e = torch.tensor(environment_output, dtype=torch.float32).reshape(-1)
            if e.shape[0] != cfg.environment_dim:
                raise ValueError(
                    f"environment_output has dim {e.shape[0]}, "
                    f"expected environment_dim={cfg.environment_dim}."
                )
            parts.append(e)
            components["environment_output"] = e.clone()

        if cfg.use_raw_physical_state:
            if physical_state is None:
                raise ValueError(
                    "physical_state is required when use_raw_physical_state=True."
                )
            ps = physical_state.detach().reshape(-1).float()
            if ps.shape[0] != cfg.physical_state_dim:
                raise ValueError(
                    f"physical_state has dim {ps.shape[0]}, "
                    f"expected physical_state_dim={cfg.physical_state_dim}."
                )
            parts.append(ps)
            components["physical_state"] = ps.clone()

        if not parts:
            raise ValueError("No state components were collected.")

        vector = torch.cat(parts)
        return RLState(vector=vector, components=components, timestamp=timestamp)

    def build_from_decision_output(
        self,
        pooled_output: torch.Tensor,
        decision_output: "PhotonShieldDecisionOutput",  # type: ignore[name-defined]
        physical_state: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None,
    ) -> RLState:
        """Convenience wrapper accepting a PhotonShieldDecisionOutput directly.

        Uses continuous fields (probabilities) rather than binary decisions.
        """
        env_out: Optional[Union[List[float], torch.Tensor]] = None
        if self.config.use_environment:
            raw = decision_output.environmental_assessment
            if isinstance(raw, (list, tuple)):
                env_out = list(raw)
            elif isinstance(raw, torch.Tensor):
                env_out = raw
            elif isinstance(raw, (int, float)):
                env_out = [float(raw)]
            else:
                env_out = [0.0] * self.config.environment_dim

        return self.build(
            pooled_output=pooled_output,
            target_probability=decision_output.target_probability,
            anomaly_probability=decision_output.anomaly_probability,
            environment_output=env_out,
            physical_state=physical_state,
            timestamp=timestamp,
        )
