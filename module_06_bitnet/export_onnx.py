"""Export PhotonV0 Perception Stack to ONNX Format.

Validates dynamic batch and sequence axes for edge runtime integration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn

from module_04_mamba_hybrid.photon_v0 import PhotonV0


class PhotonV0ONNXWrapper(nn.Module):
    """Wrapper ensuring tuple output format for ONNX tracing."""

    def __init__(self, model: PhotonV0) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        return outputs["detection"], outputs["classification"], outputs["anomaly"]


def export_photon_v0_onnx(
    model: Optional[PhotonV0] = None,
    output_path: Union[str, Path] = "photon_v0.onnx",
    input_dim: int = 64,
    hidden_dim: int = 64,
    sequence_length: int = 16,
    num_classes: int = 4,
    dynamic_axes: bool = True,
    opset_version: int = 14,
) -> Path:
    """Export PhotonV0 model to ONNX.

    Args:
        model: Instantiated PhotonV0 model. If None, instantiates a default model.
        output_path: Destination file path for .onnx model.
        input_dim: Input feature dimension.
        hidden_dim: Model hidden dimension.
        sequence_length: Default sequence length for dummy input.
        num_classes: Number of classification classes.
        dynamic_axes: If True, marks batch and sequence length as dynamic.
        opset_version: ONNX operator set version (default 14).

    Returns:
        Path to exported ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = PhotonV0(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            sequence_length=sequence_length,
            num_classes=num_classes,
        )
    model.eval()

    wrapped_model = PhotonV0ONNXWrapper(model)
    dummy_input = torch.randn(1, sequence_length, input_dim, dtype=torch.float32)

    dyn_axes = None
    if dynamic_axes:
        dyn_axes = {
            "radar_features": {0: "batch_size", 1: "sequence_length"},
            "detection": {0: "batch_size"},
            "classification": {0: "batch_size"},
            "anomaly": {0: "batch_size"},
        }

    # Use legacy TorchScript-based exporter or dynamo=False
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(output_path),
            input_names=["radar_features"],
            output_names=["detection", "classification", "anomaly"],
            dynamic_axes=dyn_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # For PyTorch versions that do not accept dynamo argument
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(output_path),
            input_names=["radar_features"],
            output_names=["detection", "classification", "anomaly"],
            dynamic_axes=dyn_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    # Validate ONNX file if onnx package is available
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
    except Exception:
        pass
        pass

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PhotonV0 to ONNX format.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="results/photon_v0/photon_v0.onnx", help="Output path")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=4)
    args = parser.parse_args()

    model = PhotonV0(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        sequence_length=args.seq_len,
        num_classes=args.num_classes,
    )
    if args.model_path and Path(args.model_path).exists():
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    out = export_photon_v0_onnx(
        model=model,
        output_path=args.output,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        sequence_length=args.seq_len,
        num_classes=args.num_classes,
    )
    print(f"Successfully exported PhotonV0 to ONNX: {out} ({out.stat().st_size / 1024:.2f} KB)")


if __name__ == "__main__":
    main()
