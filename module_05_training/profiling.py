"""Training/inference profiling utilities for Module 5."""

from __future__ import annotations
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def profile_model(
    engine: nn.Module,
    head: nn.Module,
    sample_input: Dict[str, Any],
    device: torch.device,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> Dict[str, Any]:
    """Profile parameter count, model size, latency, and throughput.

    Args:
        engine: Module 4 engine.
        head: Task head.
        sample_input: A single Module 3 output dict (unbatched, will be unsqueezed).
        device: Device to run profiling on.
        n_warmup: Number of warmup forward passes.
        n_runs: Number of timed forward passes.

    Returns:
        Dict of profiling metrics.
    """
    engine = engine.to(device)
    head = head.to(device)
    engine.eval()
    head.eval()

    # Parameter counts
    engine_params = sum(p.numel() for p in engine.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    total_params = engine_params + head_params
    trainable_params = sum(
        p.numel() for p in list(engine.parameters()) + list(head.parameters())
        if p.requires_grad
    )
    model_size_mb = total_params * 4 / (1024 ** 2)  # float32 = 4 bytes

    # Batch the sample (add batch dim)
    batched = {}
    for k, v in sample_input.items():
        if isinstance(v, torch.Tensor):
            batched[k] = v.unsqueeze(0).to(device)
        else:
            batched[k] = v

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            out = engine(batched)
            _ = head(out["pooled_output"])

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = engine(batched)
            _ = head(out["pooled_output"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    mean_lat = sum(latencies) / len(latencies)
    std_lat = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
    throughput = 1000.0 / mean_lat  # samples/sec (batch_size=1)

    result: Dict[str, Any] = {
        "engine_params": engine_params,
        "head_params": head_params,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 3),
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "throughput_samples_per_sec": round(throughput, 2),
        "device": str(device),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        result["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2
        )

    return result
