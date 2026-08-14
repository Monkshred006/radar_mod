"""Normalization Layer Wrappers for Module 4.

Provides stable layer normalization wrappers compatible with variable batch sizes
and sequence lengths.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from module_04_mamba_hybrid.config import MambaHybridConfig


def get_normalization_layer(d_model: int, norm_type: str = "layernorm") -> nn.Module:
    """Factory function for normalization layers.

    Args:
        d_model: Feature dimension.
        norm_type: Type of normalization ("layernorm", "none").

    Returns:
        nn.Module normalization instance.
    """
    if norm_type.lower() == "layernorm":
        return nn.LayerNorm(d_model)
    elif norm_type.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")
