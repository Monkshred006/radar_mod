"""Physics models for Module 8 PINN + RL.

Provides an abstract PhysicsModel interface and three concrete implementations:

KinematicPhysicsModel — SYNTHETIC VERIFICATION ONLY
    Discrete kinematic constraint:
        x_{t+1} = x_t + v_t * dt
        v_{t+1} = v_t + a_t * dt
    Does NOT represent validated PhotonShield hardware dynamics.
    Residual computed via arithmetic — no autograd required.

WaveConvectionPhysicsModel — OPTIONAL, disabled by default
    Wave-convection PDE residual (adapted from PhotonPINN-Radar paper):
        R_wave = d²u/dt² + v·du/dx - κ·d²u/dx²
    Requires the state to contain a spatiotemporal field with physically
    meaningful (u, x, t, v, κ). NOT applied to the general RL state vector.
    Uses autograd for derivative computation.

NoPhysicsModel — baseline / ablation
    Always returns zero residual. Used for data-only PINN baseline (Exp B)
    and lambda_phys = 0 ablation. Distinct from RL-only (Exp A) which
    does not use a PINN at all.

PhotonPINN-Radar reference
--------------------------
The wave-convection residual is adapted from:
    "PhotonPINN-Radar: Physics-Informed Diffusion and Tracking for
     Photonic FMCW Radar" (design reference; results are simulation-based).
The paper uses this residual as a kinematically informed transport
constraint on latent range-Doppler features, not a direct Maxwell-equation
solver. The PhotonShield adaptation retains this interpretation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from module_08_pinn_rl.config import PhysicsConfig


class PhysicsModel(ABC):
    """Abstract interface for physics residual models.

    All concrete implementations must override `residual()`.
    """

    @abstractmethod
    def residual(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state_pred: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the physics residual given a predicted state transition.

        Parameters
        ----------
        state : Tensor[B, state_dim] or Tensor[state_dim]
            Current state.
        action : Tensor[B, action_dim] or Tensor[action_dim]
            Action taken (encoded; discrete actions one-hot).
        next_state_pred : Tensor[B, state_dim]
            Predicted next state from the dynamics model.

        Returns
        -------
        Tensor
            Physics residual. Loss is typically computed as mean(residual²).
        """


class KinematicPhysicsModel(PhysicsModel):
    """Discrete kinematic physics residual — SYNTHETIC VERIFICATION ONLY.

    Governing equations (discrete-time):
        x_{t+1} = x_t + v_t * dt                (position update)
        v_{t+1} = v_t + a_t * dt                (velocity update)

    Residuals:
        r_x = x_{t+1,pred} - (x_t + v_t * dt)
        r_v = v_{t+1,pred} - (v_t + a_force * dt)

    where a_force is extracted from the action tensor. For a discrete
    action encoded as one-hot, the force is the dot product with the
    force_map tensor.

    IMPORTANT
    ---------
    This model is for SOFTWARE VERIFICATION of the PINN + RL pipeline.
    It does NOT represent validated PhotonShield hardware dynamics.
    The real physical model is application-dependent.

    Parameters
    ----------
    config : PhysicsConfig
        Provides dt, position_indices, velocity_indices.
    force_map : List[float], optional
        Maps discrete action index → scalar force. Default: [-1, 0, +1].
    """

    def __init__(
        self,
        config: PhysicsConfig,
        force_map: Optional[List[float]] = None,
    ) -> None:
        self.config = config
        self.dt = config.dt
        self.pos_idx = config.position_indices
        self.vel_idx = config.velocity_indices
        # Map from one-hot action to force value
        if force_map is not None:
            self._force_map = torch.tensor(force_map, dtype=torch.float32)
        else:
            self._force_map = None   # Falls back to using first action dim as force

    def residual(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state_pred: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute kinematic residual.

        Returns
        -------
        Tensor[B, 2]
            Stack of position and velocity residuals.
        """
        state = state.float()
        action = action.float()
        next_state_pred = next_state_pred.float()

        # Handle batched [B, dim] and unbatched [dim]
        if state.ndim == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            next_state_pred = next_state_pred.unsqueeze(0)

        x_t = state[:, self.pos_idx]           # [B, n_pos]
        v_t = state[:, self.vel_idx]           # [B, n_vel]
        x_next = next_state_pred[:, self.pos_idx]
        v_next = next_state_pred[:, self.vel_idx]

        # Extract force from action
        if self._force_map is not None:
            fm = self._force_map.to(action.device)
            # action: [B, n_actions] one-hot OR [B, 1] force
            if action.shape[-1] == fm.shape[0]:
                a_force = (action * fm.unsqueeze(0)).sum(dim=-1, keepdim=True)
            else:
                a_force = action[:, :1]
        else:
            a_force = action[:, :1]          # first dim as direct force

        # Kinematic residuals
        r_x = x_next - (x_t + v_t * self.dt)
        r_v = v_next - (v_t + a_force * self.dt)

        return torch.cat([r_x, r_v], dim=-1)  # [B, n_pos + n_vel]


class WaveConvectionPhysicsModel(PhysicsModel):
    """Wave-convection PDE residual — OPTIONAL, disabled by default.

    Adapted from the PhotonPINN-Radar design reference paper:
        R_wave = d²u/dt² + v·du/dx - κ·d²u/dx²

    Interpretation (from paper):
        u: latent range-Doppler field or scalar field of interest
        x: spatial coordinate (range dimension)
        t: temporal coordinate
        v: advection velocity (wave_velocity in PhysicsConfig)
        κ: diffusion coefficient (wave_diffusion in PhysicsConfig)

    REQUIREMENTS for use
    --------------------
    The state MUST contain a spatiotemporal field that is physically
    consistent with (u, x, t, v, κ). Do NOT apply to the general
    RL state vector.

    Dimensional consistency must be validated before enabling this model.

    Usage
    -----
    This model operates differently from KinematicPhysicsModel.
    It expects a differentiable field function  u_fn(x, t) → Tensor
    and uses autograd to compute PDE derivatives.
    Use `compute_pde_residual` for the autograd computation.
    The `residual()` method is provided for interface compatibility
    but requires passing the field function via kwargs.

    Parameters
    ----------
    config : PhysicsConfig
        Provides wave_velocity (v) and wave_diffusion (κ).
    """

    def __init__(self, config: PhysicsConfig) -> None:
        self.config = config
        self.v = config.wave_velocity
        self.kappa = config.wave_diffusion

    def compute_pde_residual(
        self,
        u_fn: "Callable",
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute R_wave via autograd.

        Parameters
        ----------
        u_fn : Callable (x: Tensor[N,1], t: Tensor[N,1]) → Tensor[N,1]
            Differentiable field function.
        x : Tensor[N, 1], requires_grad=True
            Spatial coordinates.
        t : Tensor[N, 1], requires_grad=True
            Temporal coordinates.

        Returns
        -------
        Tensor[N, 1]
            PDE residual R_wave = d²u/dt² + v·du/dx - κ·d²u/dx²
        """
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        if not t.requires_grad:
            t = t.detach().requires_grad_(True)

        u = u_fn(x, t)

        du_dx = torch.autograd.grad(
            u.sum(), x, create_graph=True, retain_graph=True
        )[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx.sum(), x, create_graph=True, retain_graph=True
        )[0]
        du_dt = torch.autograd.grad(
            u.sum(), t, create_graph=True, retain_graph=True
        )[0]
        d2u_dt2 = torch.autograd.grad(
            du_dt.sum(), t, create_graph=True, retain_graph=True
        )[0]

        return d2u_dt2 + self.v * du_dx - self.kappa * d2u_dx2

    def residual(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state_pred: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Interface-compatible residual.

        For full PDE residual computation, use `compute_pde_residual()`
        with a differentiable field function.

        This method returns zero unless a `u_fn`, `x`, and `t` are
        provided in kwargs, in which case it delegates to
        `compute_pde_residual`.
        """
        u_fn = kwargs.get("u_fn")
        x = kwargs.get("x")
        t = kwargs.get("t")
        if u_fn is not None and x is not None and t is not None:
            return self.compute_pde_residual(u_fn, x, t)
        return torch.zeros(1, dtype=torch.float32)


class NoPhysicsModel(PhysicsModel):
    """Null physics model — always returns zero residual.

    Used for:
    - Data-only PINN training (Experiment B, lambda_phys = 0)
    - Ablation baseline

    IMPORTANT: This is NOT the same as the RL-only baseline (Experiment A),
    which does not use a PINN dynamics model at all.
    """

    def residual(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state_pred: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return torch.zeros(1, dtype=state.dtype, device=state.device)


def build_physics_model(config: PhysicsConfig) -> PhysicsModel:
    """Factory function: create the appropriate PhysicsModel from config."""
    if config.physics_model == "kinematic":
        return KinematicPhysicsModel(config)
    elif config.physics_model == "wave_convection":
        return WaveConvectionPhysicsModel(config)
    elif config.physics_model == "none":
        return NoPhysicsModel()
    else:
        raise ValueError(f"Unknown physics_model: '{config.physics_model}'.")
