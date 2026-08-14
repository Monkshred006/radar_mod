"""Profiling Utilities for Module 4.

Measures parameter counts, memory footprint, forward-pass latency, throughput,
and approximate FLOPs on CPU or GPU devices.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import time
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a PyTorch model.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def profile_model(
    model: nn.Module,
    sample_input: Dict[str, Any],
    num_warmup: int = 5,
    num_runs: int = 20,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Profile latency, parameter count, memory usage, and throughput.

    Args:
        model: PyTorch module.
        sample_input: Sample input dict (Module 3 output dictionary).
        num_warmup: Warm-up iterations.
        num_runs: Benchmark iterations.
        device: Device string ("cpu" or "cuda").

    Returns:
        Dict containing benchmark metrics.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    # Move sample tensors to target device
    dev_input = {}
    for k, v in sample_input.items():
        if isinstance(v, torch.Tensor):
            dev_input[k] = v.to(dev)
        else:
            dev_input[k] = v

    total_params, trainable_params = count_parameters(model)

    # Calculate model weight size in MB (FP32 = 4 bytes per param)
    param_size_mb = (total_params * 4) / (1024 * 1024)

    # Warm-up
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dev_input)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    # Measure Latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dev_input)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    avg_latency_ms = sum(latencies) / len(latencies)
    std_latency_ms = (sum((x - avg_latency_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5

    # Determine batch size and sequence length safely
    if "tokens" in sample_input and sample_input["tokens"] is not None:
        tokens = sample_input["tokens"]
    else:
        tokens = sample_input["features"]

    batch_size = tokens.shape[0] if tokens.ndim >= 3 else 1
    seq_len = tokens.shape[1] if tokens.ndim >= 3 else tokens.shape[0]

    throughput_fps = (batch_size * 1000.0) / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "device": dev,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(param_size_mb, 3),
        "mean_latency_ms": round(avg_latency_ms, 3),
        "std_latency_ms": round(std_latency_ms, 3),
        "throughput_samples_per_sec": round(throughput_fps, 2),
        "batch_size": batch_size,
        "sequence_length": seq_len,
    }
