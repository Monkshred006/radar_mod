"""Ternary Weight Quantization, STE Autograd, and Packing Subsystem.

Defines:
  1. TernarySTEFunction: Autograd function using Straight-Through Estimator (STE).
     Forward pass outputs scaled ternary weights W_quant ∈ {-α, 0, +α}.
     Backward pass passes gradients straight through to FP32 master weights.
  2. round_to_ternary: Functional wrapper for ternary quantization.
  3. pack_ternary / unpack_ternary: Base-3 packing (5 ternary values per byte).
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
from module_06_bitnet.scaling import compute_weight_scale


class TernarySTEFunction(torch.autograd.Function):
    """Straight-Through Estimator (STE) for Ternary Weight Quantization.

    Mathematical Formulation:
      Forward:
        S = compute_weight_scale(W)
        W_normalized = W / S
        W_ternary = clip(round(W_normalized), -1, +1)
        W_quantized = S * W_ternary

      Backward (STE Surrogate Gradient):
        ∂L/∂W = ∂L/∂W_quantized
        (The non-differentiable round() operation is bypassed during backward pass)
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.FunctionCtx,
        w: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(w, scale)
        # Normalize and round to ternary {-1, 0, +1}
        w_normalized = w / scale
        w_ternary = torch.clamp(torch.round(w_normalized), min=-1.0, max=1.0)
        w_quantized = w_ternary * scale
        return w_quantized

    @staticmethod
    def backward(
        ctx: torch.autograd.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        # Straight-Through Estimator: pass gradient un-altered to master weights
        return grad_output, None


def round_to_ternary(
    w: torch.Tensor,
    scale_method: str = "mean_abs",
    scale_scope: str = "per_tensor",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize FP32 weight tensor W to ternary representation using STE.

    Args:
        w: Master FP32 weight tensor.
        scale_method: 'mean_abs' or 'max_abs'.
        scale_scope: 'per_tensor' or 'per_channel'.

    Returns:
        Tuple of (w_quantized, scale, w_ternary_int):
          - w_quantized: Scaled ternary weights tensor (autograd enabled).
          - scale: Calculated scale factor S.
          - w_ternary_int: Unscaled discrete ternary symbols in {-1, 0, 1} (int8).
    """
    scale = compute_weight_scale(w, method=scale_method, scope=scale_scope)
    w_quantized = TernarySTEFunction.apply(w, scale)

    # Detached unscaled discrete ternary integer values
    with torch.no_grad():
        w_norm = w / scale
        w_ternary_int = torch.clamp(torch.round(w_norm), min=-1, max=1).to(torch.int8)

    return w_quantized, scale, w_ternary_int


# ──────────────────────────────────────────────────────────────────────────────
# Optional Base-3 Packing Subsystem
# ──────────────────────────────────────────────────────────────────────────────

def pack_ternary(w_ternary: torch.Tensor) -> bytes:
    """Pack ternary tensor in {-1, 0, 1} into compact byte representation.

    Base-3 encoding: 5 ternary values fit into 1 byte (3^5 = 243 <= 256).
    Map {-1, 0, 1} -> {0, 1, 2}.

    Args:
        w_ternary: Tensor with values in {-1, 0, 1} (or float -1.0, 0.0, 1.0).

    Returns:
        Packed bytes object.
    """
    flat = w_ternary.detach().cpu().to(torch.int8).reshape(-1)
    # Shift {-1, 0, 1} -> {0, 1, 2}
    shifted = (flat + 1).to(torch.uint8)

    n = shifted.numel()
    byte_list = bytearray()

    for i in range(0, n, 5):
        chunk = shifted[i:i + 5]
        val = 0
        multiplier = 1
        for symbol in chunk:
            val += int(symbol.item()) * multiplier
            multiplier *= 3
        byte_list.append(val)

    return bytes(byte_list)


def unpack_ternary(packed: bytes, original_shape: torch.Size) -> torch.Tensor:
    """Unpack compact byte representation back into discrete ternary tensor {-1, 0, 1}.

    Args:
        packed: Packed bytes from pack_ternary.
        original_shape: Desired output tensor shape.

    Returns:
        Tensor of shape original_shape with values in {-1, 0, 1} (dtype int8).
    """
    total_elements = 1
    for s in original_shape:
        total_elements *= s

    unpacked_list = []
    for byte_val in packed:
        val = int(byte_val)
        for _ in range(5):
            if len(unpacked_list) < total_elements:
                symbol = val % 3
                val = val // 3
                unpacked_list.append(symbol - 1)  # Shift {0, 1, 2} -> {-1, 0, 1}

    result = torch.tensor(unpacked_list, dtype=torch.int8).reshape(original_shape)
    return result
