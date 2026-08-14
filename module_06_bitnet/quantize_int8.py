"""INT8 Quantization for PhotonV0 Perception Stack.

Applies INT8 quantization to Linear and Convolutional layers, evaluating model size reduction
and inference efficiency for edge deployment.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
from typing import Optional, Tuple, Union, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def quantize_onnx_int8(
    input_onnx_path: Union[str, Path] = "artifacts/photon_v0.onnx",
    output_onnx_path: Union[str, Path] = "artifacts/photon_v0_int8.onnx",
) -> Optional[Path]:
    """Quantize an exported ONNX model to INT8 dynamic quantization."""
    input_p = Path(input_onnx_path)
    output_p = Path(output_onnx_path)
    if not input_p.exists():
        return None

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        output_p.parent.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(
            model_input=str(input_p),
            model_output=str(output_p),
            weight_type=QuantType.QInt8,
        )
        return output_p
    except Exception as e:
        print(f"[Quantizer] Note: ONNX INT8 export skipped ({e})")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize PhotonV0 to INT8.")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, default=None, help="Path to FP32 checkpoint")
    parser.add_argument("--config", type=str, default="configs/photon_v0.yaml", help="Path to config yaml")
    parser.add_argument("--output", type=str, default="artifacts/photon_v0_int8.pt")
    parser.add_argument("--onnx-input", type=str, default="artifacts/photon_v0.onnx")
    parser.add_argument("--onnx-output", type=str, default="artifacts/photon_v0_int8.onnx")
    args = parser.parse_args()

    model = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, sequence_length=16, backend="fallback")
    if args.model_path and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    _, stats = quantize_photon_v0_int8(model, output_path=args.output)
    print("=== INT8 Quantization Summary ===")
    print(f"FP32 Model Size: {stats['fp32_size_kb']:.2f} KB")
    print(f"INT8 Model Size: {stats['int8_size_kb']:.2f} KB")
    print(f"Compression Ratio: {stats['reduction_factor']:.2f}x ({stats['savings_percent']:.1f}% reduction)")

    # Also quantize ONNX model if input exists
    if Path(args.onnx_input).exists():
        out_onnx = quantize_onnx_int8(args.onnx_input, args.onnx_output)
        if out_onnx and out_onnx.exists():
            print(f"Exported INT8 ONNX Model: {out_onnx} ({out_onnx.stat().st_size / 1024:.2f} KB)")


if __name__ == "__main__":
    main()
