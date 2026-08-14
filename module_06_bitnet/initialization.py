"""Initialization strategies for BitNet models.

Supports:
  1. 'fp32_converted': Initialize BitLinear master weights from pre-trained FP32 weights.
  2. 'fresh': Initialize master weights from scratch (Kaiming Uniform).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from module_06_bitnet.bit_linear import BitLinear


def initialize_bitnet_weights(
    bitnet_model: nn.Module,
    mode: str = "fp32_converted",
    source_fp32_model: Optional[nn.Module] = None,
) -> None:
    """Initialize BitLinear layers in bitnet_model.

    Args:
        bitnet_model: Model containing BitLinear layers.
        mode: 'fp32_converted' or 'fresh'.
        source_fp32_model: Source FP32 model (required if mode is 'fp32_converted').
    """
    if mode == "fp32_converted":
        if source_fp32_model is None:
            raise ValueError("source_fp32_model must be provided when mode is 'fp32_converted'")

        # Copy state dict matching keys
        fp32_state = source_fp32_model.state_dict()
        bitnet_state = bitnet_model.state_dict()

        for k, v in fp32_state.items():
            if k in bitnet_state and bitnet_state[k].shape == v.shape:
                bitnet_state[k].copy_(v)

    elif mode == "fresh":
        for module in bitnet_model.modules():
            if isinstance(module, BitLinear):
                module.reset_parameters()
    else:
        raise ValueError(f"Unknown initialization mode: '{mode}'. Choose from 'fp32_converted', 'fresh'")
