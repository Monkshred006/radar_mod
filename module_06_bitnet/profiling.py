"""Profiling & Efficiency Accounting Utilities for Module 6.

Measures parameter counts, theoretical weight storage (log2(3) bits/weight),
actual serialized checkpoint size, CPU/CUDA latency (mean, p50, p95),
throughput, and peak memory.
"""

from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn

from module_06_bitnet.bit_linear import BitLinear
from module_06_bitnet.ternary import pack_ternary
from module_06_bitnet.hardware_backend import HARDWARE_DISCLAIMER


def profile_bitnet_model(
    engine: nn.Module,
    head: nn.Module,
    sample_input: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    packing_enabled: bool = False,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> Dict[str, Any]:
    """Profile BitNet model efficiency, theoretical storage, actual latency, and memory.

    Args:
        engine: BitNet Module 4 engine.
        head: BitNet task head.
        sample_input: Module 3 sample dict (unbatched, will be unsqueezed).
        checkpoint_path: Path to serialized checkpoint file on disk.
        packing_enabled: Whether packing is enabled.
        n_warmup: Warmup forward passes before timing.
        n_runs: Timed forward passes.

    Returns:
        Dict of profiling metrics.
    """
    combined = nn.ModuleList([engine, head])
    device = next(combined.parameters()).device
    combined.eval()

    # 1. Parameter Accounting
    total_params = 0
    ternary_params = 0
    fp32_params = 0

    for name, module in combined.named_modules():
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if isinstance(module, BitLinear):
            ternary_params += n_params
            total_params += n_params
        elif isinstance(module, nn.Linear):
            fp32_params += n_params
            total_params += n_params
        else:
            fp32_params += n_params
            total_params += n_params

    pct_ternary = (ternary_params / total_params * 100.0) if total_params > 0 else 0.0
    pct_fp32 = 100.0 - pct_ternary

    # 2. Weight Storage Accounting
    # Theoretical ternary weight capacity: log2(3) ≈ 1.585 bits/weight
    theoretical_ternary_bits = ternary_params * 1.5849625
    theoretical_fp32_bits = fp32_params * 32.0
    theoretical_total_bytes = (theoretical_ternary_bits + theoretical_fp32_bits) / 8.0
    theoretical_avg_bits_per_weight = (theoretical_ternary_bits + theoretical_fp32_bits) / max(total_params, 1)

    # Actual serialized file size
    actual_checkpoint_bytes = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        actual_checkpoint_bytes = Path(checkpoint_path).stat().st_size
    actual_checkpoint_mb = actual_checkpoint_bytes / (1024 ** 2)

    # Packed ternary size calculation
    packed_bytes = 0
    if packing_enabled:
        for module in combined.modules():
            if isinstance(module, BitLinear):
                packed_bytes += len(pack_ternary(module.weight.data))

    # 3. Latency & Throughput Benchmarking
    batched = {}
    for k, v in sample_input.items():
        if isinstance(v, torch.Tensor):
            batched[k] = v.unsqueeze(0).to(device)
        else:
            batched[k] = v

    with torch.no_grad():
        for _ in range(n_warmup):
            out = engine(batched)
            _ = head(out["pooled_output"])

    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = engine(batched)
            _ = head(out["pooled_output"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    lat_np = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_np))
    median_lat = float(np.median(lat_np))
    p95_lat = float(np.percentile(lat_np, 95))
    throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    peak_gpu_mem_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_gpu_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

    return {
        "total_params": total_params,
        "ternary_params": ternary_params,
        "fp32_params": fp32_params,
        "pct_ternary": round(pct_ternary, 2),
        "pct_fp32": round(pct_fp32, 2),
        "theoretical_bits_per_weight": round(theoretical_avg_bits_per_weight, 2),
        "theoretical_weight_storage_bytes": int(theoretical_total_bytes),
        "theoretical_weight_storage_mb": round(theoretical_total_bytes / (1024 ** 2), 3),
        "actual_checkpoint_bytes": actual_checkpoint_bytes,
        "actual_checkpoint_mb": round(actual_checkpoint_mb, 3),
        "packing_enabled": packing_enabled,
        "packed_bytes": packed_bytes,
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_samples_per_sec": round(throughput, 2),
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "device": str(device),
        "hardware_disclaimer": HARDWARE_DISCLAIMER,
    }
