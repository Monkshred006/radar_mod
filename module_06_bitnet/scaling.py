"""Weight and Activation Scaling Utilities for Module 6.

Provides configurable weight scale calculation:
  - Methods: 'mean_abs' (mean-absolute-weight), 'max_abs' (max-absolute-weight)
  - Scopes:  'per_tensor', 'per_channel'
"""

from __future__ import annotations
import torch


def compute_weight_scale(
    w: torch.Tensor,
    method: str = "mean_abs",
    scope: str = "per_tensor",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute scaling factor S for weight matrix W.

    Args:
        w: Weight tensor of shape [out_features, in_features] or arbitrary shape.
        method: 'mean_abs' (mean-absolute-weight) or 'max_abs'.
        scope: 'per_tensor' (scalar scale) or 'per_channel' (vector scale along dim 0).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Scale tensor S matching scope geometry.
    """
    if scope == "per_channel" and w.ndim >= 2:
        # Scale per output channel (dim 0)
        reduce_dims = tuple(range(1, w.ndim))
        if method == "mean_abs":
            s = torch.mean(torch.abs(w), dim=reduce_dims, keepdim=True)
        elif method == "max_abs":
            s = torch.amax(torch.abs(w), dim=reduce_dims, keepdim=True)
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    else:
        # Per-tensor scale
        if method == "mean_abs":
            s = torch.mean(torch.abs(w))
        elif method == "max_abs":
            s = torch.amax(torch.abs(w))
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    # Clamp minimum scale to eps to avoid division by zero
    s = torch.clamp(s, min=eps)
    return s


def compute_activation_scale(
    x: torch.Tensor,
    method: str = "max_abs",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute scaling factor for activation tensor X.

    Args:
        x: Activation tensor [B, ..., D].
        method: 'max_abs' or 'mean_abs'.
        eps: Epsilon to prevent zero division.

    Returns:
        Scalar activation scale tensor.
    """
    if method == "max_abs":
        s = torch.amax(torch.abs(x))
    elif method == "mean_abs":
        s = torch.mean(torch.abs(x))
    else:
        raise ValueError(f"Unknown activation scaling method: {method}")

    return torch.clamp(s, min=eps)
