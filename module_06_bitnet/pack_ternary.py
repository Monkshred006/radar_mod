"""Ternary {-1, 0, +1} Weight Quantization & 2-Bit Array Packing.

Converts full-precision weights to ternary BitNet 1.58b representations and packs 4 weights
into each uint8 byte for ultra-compact embedded C++ microcontroller execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, Any
import numpy as np
import torch
import torch.nn as nn

from module_04_mamba_hybrid.photon_v0 import PhotonV0


def quantize_to_ternary(weight: torch.Tensor, threshold_ratio: float = 0.5) -> Tuple[torch.Tensor, float]:
    """Quantize a 2D weight matrix to ternary {-1, 0, +1} and scaling factor alpha.

    Args:
        weight: Float tensor `[Out, In]`.
        threshold_ratio: Scaling for deadzone threshold.

    Returns:
        Tuple of:
            - ternary_tensor: `{-1, 0, 1}` tensor of same shape.
            - alpha: Float scaling factor.
    """
    scale = torch.mean(torch.abs(weight)).item()
    if scale < 1e-8:
        return torch.zeros_like(weight), 1.0

    threshold = threshold_ratio * scale
    ternary = torch.zeros_like(weight)
    ternary[weight > threshold] = 1.0
    ternary[weight < -threshold] = -1.0

    # Optimal scaling factor alpha = sum(|W_ternary * W_orig|) / sum(W_ternary^2)
    non_zero = (ternary != 0)
    if non_zero.sum() > 0:
        alpha = torch.mean(torch.abs(weight[non_zero])).item()
    else:
        alpha = scale

    return ternary, alpha


def pack_ternary_matrix_to_uint8(ternary_matrix: np.ndarray) -> np.ndarray:
    """Pack a ternary {-1, 0, +1} numpy array into 2-bit values packed 4 per uint8 byte.

    Encoding:
        0  -> 0b00 (0)
        +1 -> 0b01 (1)
        -1 -> 0b10 (2)
        pad-> 0b00 (0)
    """
    flat = ternary_matrix.flatten()
    n = len(flat)

    # Pad to multiple of 4
    pad_len = (4 - (n % 4)) % 4
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), constant_values=0)

    # Map to 2-bit codes: 0 -> 0, +1 -> 1, -1 -> 2
    code_map = np.zeros(len(flat), dtype=np.uint8)
    code_map[flat > 0.5] = 1
    code_map[flat < -0.5] = 2

    # Pack 4 consecutive codes into 1 uint8 byte: (c0 << 6) | (c1 << 4) | (c2 << 2) | c3
    reshaped = code_map.reshape(-1, 4)
    packed = (
        (reshaped[:, 0] << 6)
        | (reshaped[:, 1] << 4)
        | (reshaped[:, 2] << 2)
        | (reshaped[:, 3])
    ).astype(np.uint8)

    return packed


def pack_model_ternary(
    model: PhotonV0,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convert model linear weights to packed ternary format and save binary artifact.

    Args:
        model: PhotonV0 model.
        output_path: Optional path to save packed .npz or .bin file.

    Returns:
        Dictionary of quantization metadata and compression statistics.
    """
    model.eval()
    packed_weights = {}
    scales = {}
    shapes = {}

    total_orig_bytes = 0
    total_packed_bytes = 0

    for name, param in model.named_parameters():
        if "weight" in name and param.ndim >= 2:
            w = param.detach().cpu()
            ternary, alpha = quantize_to_ternary(w)
            packed = pack_ternary_matrix_to_uint8(ternary.numpy())

            packed_weights[name] = packed
            scales[name] = float(alpha)
            shapes[name] = list(w.shape)

            orig_bytes = w.numel() * 4  # FP32 = 4 bytes
            packed_bytes = len(packed)   # Packed = 1 byte per 4 weights (2 bits/weight)

            total_orig_bytes += orig_bytes
            total_packed_bytes += packed_bytes

    compression_ratio = total_orig_bytes / max(total_packed_bytes, 1)

    result = {
        "total_orig_fp32_kb": total_orig_bytes / 1024.0,
        "total_packed_kb": total_packed_bytes / 1024.0,
        "compression_ratio": compression_ratio,
        "num_layers_packed": len(packed_weights),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(output_path),
            **{f"w_{k}": v for k, v in packed_weights.items()},
            scales=scales,
            shapes=shapes,
            stats=result,
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack PhotonV0 weights to ternary 2-bit binary arrays.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to FP32 checkpoint")
    parser.add_argument("--output", type=str, default="results/photon_v0/photon_v0_ternary.npz")
    args = parser.parse_args()

    model = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16)
    if args.model_path and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    stats = pack_model_ternary(model, output_path=args.output)
    print("=== Ternary {-1, 0, +1} BitNet Packing Summary ===")
    print(f"FP32 Weight Size: {stats['total_orig_fp32_kb']:.2f} KB")
    print(f"Packed Ternary (2-bit) Size: {stats['total_packed_kb']:.2f} KB")
    print(f"Compression Ratio: {stats['compression_ratio']:.2f}x (~16x vs FP32)")


if __name__ == "__main__":
    main()
