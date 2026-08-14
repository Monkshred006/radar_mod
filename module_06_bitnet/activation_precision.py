"""Activation Precision Utilities for Module 6.

Supports configurable activation precision modes:
  - 'fp32' (default for Module 6 research baseline)
  - 'fp16'
  - 'bf16'
  - 'int8' (simulated 8-bit dynamic quantization)
"""

from __future__ import annotations
import torch
from module_06_bitnet.scaling import compute_activation_scale


def apply_activation_precision(
    x: torch.Tensor,
    precision: str = "fp32",
) -> torch.Tensor:
    """Apply precision conversion / dynamic quantization to activation tensor X.

    Args:
        x: Input activation tensor.
        precision: 'fp32', 'fp16', 'bf16', or 'int8'.

    Returns:
        Activation tensor in requested precision representation.
    """
    mode = precision.lower()

    if mode == "fp32":
        return x.to(torch.float32)
    elif mode == "fp16":
        return x.to(torch.float16).to(x.dtype)
    elif mode == "bf16":
        return x.to(torch.bfloat16).to(x.dtype)
    elif mode == "int8":
        # Simulated INT8 dynamic quantization with STE / round-dequantize
        scale = compute_activation_scale(x, method="max_abs")
        scaled_x = x / scale * 127.0
        clipped_x = torch.clamp(torch.round(scaled_x), min=-128.0, max=127.0)
        dequantized_x = (clipped_x / 127.0) * scale
        return dequantized_x
    else:
        raise ValueError(
            f"Unsupported activation_precision: '{precision}'. "
            "Choose from: 'fp32', 'fp16', 'bf16', 'int8'"
        )
