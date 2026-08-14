"""Hardware Backend Abstraction & Disclaimer Utilities for Module 6."""

from __future__ import annotations
from typing import Dict, Any
import torch


HARDWARE_DISCLAIMER = (
    "Low-bit model validated at the neural-model level; "
    "dedicated ternary hardware acceleration has not yet been validated."
)


def get_hardware_backend_info(backend_name: str = "pytorch") -> Dict[str, Any]:
    """Return hardware backend capabilities and honesty disclaimers.

    Args:
        backend_name: 'pytorch', 'cuda', or 'edge_stub'.

    Returns:
        Dict describing hardware capabilities and runtime disclaimer.
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    return {
        "backend": backend_name,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "disclaimer": HARDWARE_DISCLAIMER,
        "supports_native_ternary_hardware_ops": False,  # PyTorch uses FP32/INT8 execution
    }
