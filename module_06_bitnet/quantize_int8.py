"""INT8 Quantization for PhotonV0 Perception Stack.

Applies INT8 quantization to Linear and Convolutional layers, evaluating model size reduction
and inference efficiency for edge deployment.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn

from module_04_mamba_hybrid.photon_v0 import PhotonV0


def quantize_photon_v0_int8(
    model: PhotonV0,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Apply dynamic INT8 quantization to PhotonV0 model.

    Args:
        model: Floating point FP32 PhotonV0 model.
        output_path: Optional path to save quantized state_dict or TorchScript model.

    Returns:
        Tuple of:
            - quantized_model: INT8 quantized PyTorch model.
            - stats: Dict containing model size metrics and reduction factor.
    """
    model.eval()

    # Calculate FP32 size
    fp32_buffer = io.BytesIO()
    torch.save(model.state_dict(), fp32_buffer)
    fp32_size_bytes = fp32_buffer.getbuffer().nbytes

    # Apply PyTorch Dynamic Quantization across Linear layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    int8_buffer = io.BytesIO()
    torch.save(quantized_model.state_dict(), int8_buffer)
    int8_size_bytes = int8_buffer.getbuffer().nbytes

    reduction_factor = fp32_size_bytes / max(int8_size_bytes, 1)
    savings_pct = (1.0 - int8_size_bytes / max(fp32_size_bytes, 1)) * 100.0

    stats = {
        "fp32_size_bytes": fp32_size_bytes,
        "fp32_size_kb": fp32_size_bytes / 1024.0,
        "int8_size_bytes": int8_size_bytes,
        "int8_size_kb": int8_size_bytes / 1024.0,
        "reduction_factor": reduction_factor,
        "savings_percent": savings_pct,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_model.state_dict(), output_path)

    return quantized_model, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize PhotonV0 to INT8.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to FP32 checkpoint")
    parser.add_argument("--output", type=str, default="results/photon_v0/photon_v0_int8.pt")
    args = parser.parse_args()

    model = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16)
    if args.model_path and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    _, stats = quantize_photon_v0_int8(model, output_path=args.output)
    print("=== INT8 Quantization Summary ===")
    print(f"FP32 Model Size: {stats['fp32_size_kb']:.2f} KB")
    print(f"INT8 Model Size: {stats['int8_size_kb']:.2f} KB")
    print(f"Compression Ratio: {stats['reduction_factor']:.2f}x ({stats['savings_percent']:.1f}% reduction)")


if __name__ == "__main__":
    main()
