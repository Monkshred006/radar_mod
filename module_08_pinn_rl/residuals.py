"""Physics residual computation functions for Module 8.

Provides standalone functions for computing physics residuals that can be
called independently of the PhysicsModel class hierarchy, e.g. in tests
or custom training loops.

Kinematic residual (SYNTHETIC VERIFICATION ONLY):
    Uses arithmetic — no autograd required.
    r_x = x_{t+1,pred} - (x_t + v_t * dt)
    r_v = v_{t+1,pred} - (v_t + a * dt)

Wave-convection residual (OPTIONAL):
    Uses torch.autograd.grad.
    R_wave = d²u/dt² + v·du/dx - κ·d²u/dx²
    retain_graph is handled correctly; no silent detach in the physics path.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch


def compute_kinematic_residual(
    state: torch.Tensor,
    next_state_pred: torch.Tensor,
    action_force: torch.Tensor,
    dt: float = 0.1,
    position_indices: Tuple[int, ...] = (0,),
    velocity_indices: Tuple[int, ...] = (1,),
) -> torch.Tensor:
    """Compute the discrete kinematic physics residual.

    Governing equations (SYNTHETIC VERIFICATION ONLY):
        x_{t+1} = x_t + v_t * dt
        v_{t+1} = v_t + a_t * dt

    Parameters
    ----------
    state : Tensor[B, state_dim] or Tensor[state_dim]
        Current state containing position and velocity components.
    next_state_pred : Tensor[B, state_dim]
        Predicted next state from the dynamics model.
    action_force : Tensor[B, 1] or Tensor[1]
        Scalar acceleration/force applied in this step.
    dt : float
        Timestep.
    position_indices : tuple of int
        Indices in state that correspond to position components.
    velocity_indices : tuple of int
        Indices in state that correspond to velocity components.

    Returns
    -------
    Tensor[B, n_pos + n_vel]
        Concatenated position and velocity residuals.
        Zero residual indicates perfect kinematic consistency.
    """
    pos_idx = list(position_indices)
    vel_idx = list(velocity_indices)

    state = state.float()
    next_state_pred = next_state_pred.float()
    action_force = action_force.float()

    if state.ndim == 1:
        state = state.unsqueeze(0)
        next_state_pred = next_state_pred.unsqueeze(0)
        action_force = action_force.unsqueeze(0)

    x_t = state[:, pos_idx]
    v_t = state[:, vel_idx]
    x_next = next_state_pred[:, pos_idx]
    v_next = next_state_pred[:, vel_idx]

    r_x = x_next - (x_t + v_t * dt)
    r_v = v_next - (v_t + action_force * dt)

    return torch.cat([r_x, r_v], dim=-1)


def compute_wave_convection_residual(
    u_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    wave_velocity: float = 1.0,
    wave_diffusion: float = 0.01,
) -> torch.Tensor:
    """Compute the wave-convection PDE residual via autograd.

    Adapted from the PhotonPINN-Radar design reference paper:
        R_wave = d²u/dt² + v·du/dx - κ·d²u/dx²

    Interpretation (from paper):
        u  : latent range-Doppler field or scalar field of interest
        x  : spatial coordinate (range dimension)
        t  : temporal coordinate
        v  : advection velocity (wave_velocity)
        κ  : diffusion coefficient (wave_diffusion)

    REQUIREMENTS
    ------------
    * The field function u_fn must be differentiable (no detach in its path).
    * x and t must either have requires_grad=True already, or this function
      will set requires_grad=True internally.
    * This function uses create_graph=True so the residual can be
      differentiated w.r.t. u_fn parameters.

    Parameters
    ----------
    u_fn : Callable (x: Tensor[N,1], t: Tensor[N,1]) → Tensor[N,1]
        Differentiable function mapping spatial and temporal coordinates to
        field values. Must NOT detach x or t internally.
    x : Tensor[N, 1]
        Spatial coordinates.
    t : Tensor[N, 1]
        Temporal coordinates.
    wave_velocity : float
        Advection velocity v.
    wave_diffusion : float
        Diffusion coefficient κ. Must be >= 0.

    Returns
    -------
    Tensor[N, 1]
        Pointwise PDE residual. Loss: mean(residual²).
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

    residual = d2u_dt2 + wave_velocity * du_dx - wave_diffusion * d2u_dx2
    return residual
