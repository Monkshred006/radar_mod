"""Hardware Profiler for Arduino Uno Q Deployment.

Evaluates memory footprint, peak SRAM activation requirements, MAC operations,
and projected hardware feasibility against Arduino Uno Q hardware specifications.
"""

from __future__ import annotations

import argparse
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from module_04_mamba_hybrid.photon_v0 import PhotonV0, count_parameters
from module_06_bitnet.runtime_specs import ARDUINO_UNO_Q_PROFILE, HardwareProfile


def estimate_photon_v0_macs(
    input_dim: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 2,
    sequence_length: int = 16,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    num_classes: int = 4,
) -> int:
    """Estimate total Multiply-Accumulate (MAC) operations for one forward pass.

    Breakdown per timestep t in 1..T:
    1. Input Projection: input_dim * hidden_dim
    2. Per Mini-Mamba Layer:
       - In-projection: hidden_dim * (2 * expand * hidden_dim) = 64 * 256 = 16,384
       - 1D Depthwise Conv: (expand * hidden_dim) * d_conv = 128 * 4 = 512
       - SSM Projections: 128 * (2*16 + 1) + 1 * 128 = 4,224 + 128 = 4,352
       - SSM Recurrent Scan (per step): 128 * 16 (dA*h) + 128 * 16 (dB*x) + 128 * 16 (h*C) = 6,144
       - Out-projection: (expand * hidden_dim) * hidden_dim = 128 * 64 = 8,192
       Total per layer per timestep ~ 35,584
    3. Three Task Heads (computed once per sequence on pooled latent):
       - Detection Head: 64 * 32 + 32 * 1 = 2,080
       - Classification Head: 64 * 32 + 32 * num_classes = 2,048 + 128 = 2,176
       - Anomaly Head: 64 * 32 + 32 * 1 = 2,080
    """
    d_inner = expand * hidden_dim
    # Linear projection
    input_macs = (input_dim * hidden_dim * sequence_length) if input_dim != hidden_dim else 0

    # Mini-Mamba per layer
    macs_in_proj = hidden_dim * (2 * d_inner) * sequence_length
    macs_conv1d = d_inner * d_conv * sequence_length
    macs_ssm_proj = (d_inner * (d_state * 2 + 1) + d_inner) * sequence_length
    macs_scan = (3 * d_inner * d_state) * sequence_length
    macs_out_proj = (d_inner * hidden_dim) * sequence_length

    macs_per_layer = macs_in_proj + macs_conv1d + macs_ssm_proj + macs_scan + macs_out_proj
    total_mamba_macs = num_layers * macs_per_layer

    # Heads (evaluated on pooled state)
    macs_det = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2) * 1
    macs_cls = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2) * num_classes
    macs_ano = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2) * 1
    heads_macs = macs_det + macs_cls + macs_ano

    total_macs = input_macs + total_mamba_macs + heads_macs
    return int(total_macs)


def profile_for_uno_q(
    model: Optional[PhotonV0] = None,
    sequence_length: int = 16,
    input_dim: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 2,
    profile_spec: HardwareProfile = ARDUINO_UNO_Q_PROFILE,
) -> Dict[str, Any]:
    """Profile PhotonV0 against Arduino Uno Q hardware specifications.

    Returns:
        Comprehensive profile dictionary with sizing, memory, MACs, and feasibility checks.
    """
    if model is None:
        model = PhotonV0(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            sequence_length=sequence_length,
        )

    num_params = count_parameters(model, trainable_only=False)

    # Memory calculations
    fp32_flash_bytes = num_params * 4
    int8_flash_bytes = num_params * 1
    ternary_packed_bytes = int(num_params * 0.25)  # 2 bits = 0.25 bytes per weight

    # SRAM Activation memory (intermediate tensor buffers)
    # Peak buffer: input [T, D] + layer buffers [2 * T * d_inner] + SSM state [d_inner * d_state]
    d_inner = 2 * hidden_dim
    d_state = 16
    peak_activation_floats = (
        (sequence_length * hidden_dim)         # Input
        + (sequence_length * 2 * d_inner)      # Projection / Conv buffer
        + (d_inner * d_state)                  # Recurrent state h
        + (sequence_length * hidden_dim)       # Output buffer
    )
    peak_sram_fp32_bytes = peak_activation_floats * 4
    peak_sram_int8_bytes = peak_activation_floats * 1

    # MACs
    total_macs = estimate_photon_v0_macs(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sequence_length=sequence_length,
        d_state=d_state,
        expand=2,
    )

    # Latency estimation at clock frequency
    # Assuming ~2-3 clock cycles per MAC on integer MCU with pipelining
    cycles_per_mac = 3
    estimated_latency_ms = (total_macs * cycles_per_mac / profile_spec.clock_frequency_hz) * 1000.0

    # Feasibility verdict
    flash_fits_int8 = int8_flash_bytes < (profile_spec.flash_bytes * 0.7)  # Leave 30% for firmware
    sram_fits_int8 = peak_sram_int8_bytes < (profile_spec.sram_bytes * 0.6) # Leave 40% for stack/RTOS

    return {
        "target_hardware": profile_spec.name,
        "target_flash_kb": profile_spec.flash_bytes / 1024,
        "target_sram_kb": profile_spec.sram_bytes / 1024,
        "target_clock_mhz": profile_spec.clock_frequency_hz / 1e6,
        "parameter_count": num_params,
        "weights_fp32_kb": fp32_flash_bytes / 1024.0,
        "weights_int8_kb": int8_flash_bytes / 1024.0,
        "weights_ternary_packed_kb": ternary_packed_bytes / 1024.0,
        "peak_sram_fp32_kb": peak_sram_fp32_bytes / 1024.0,
        "peak_sram_int8_kb": peak_sram_int8_bytes / 1024.0,
        "total_macs": total_macs,
        "total_flops": total_macs * 2,
        "estimated_latency_ms": estimated_latency_ms,
        "flash_fit_int8": flash_fits_int8,
        "sram_fit_int8": sram_fits_int8,
        "overall_fit": flash_fits_int8 and sram_fits_int8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PhotonV0 for Arduino Uno Q.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    results = profile_for_uno_q(
        hidden_dim=args.hidden_dim,
        sequence_length=args.seq_len,
        num_layers=args.num_layers,
    )

    print("================================================================")
    print(f" PhotonShield AI - Hardware Profile: {results['target_hardware']}")
    print("================================================================")
    print(f"Hardware Budget: Flash: {results['target_flash_kb']:.0f} KB | SRAM: {results['target_sram_kb']:.0f} KB | Clock: {results['target_clock_mhz']:.0f} MHz")
    print("----------------------------------------------------------------")
    print(f"Total Parameters:             {results['parameter_count']:,}")
    print(f"Weights (FP32 Flash):         {results['weights_fp32_kb']:.2f} KB")
    print(f"Weights (INT8 Flash):         {results['weights_int8_kb']:.2f} KB ({results['weights_int8_kb'] / results['target_flash_kb'] * 100:.1f}% flash)")
    print(f"Weights (Ternary Packed):     {results['weights_ternary_packed_kb']:.2f} KB ({results['weights_ternary_packed_kb'] / results['target_flash_kb'] * 100:.1f}% flash)")
    print("----------------------------------------------------------------")
    print(f"Peak SRAM Activation (FP32):  {results['peak_sram_fp32_kb']:.2f} KB")
    print(f"Peak SRAM Activation (INT8):  {results['peak_sram_int8_kb']:.2f} KB ({results['peak_sram_int8_kb'] / results['target_sram_kb'] * 100:.1f}% SRAM)")
    print("----------------------------------------------------------------")
    print(f"Inference Compute:            {results['total_macs']:,} MACs ({results['total_flops']:,} FLOPs)")
    print(f"Estimated MCU Latency:        ~{results['estimated_latency_ms']:.2f} ms per frame")
    print("----------------------------------------------------------------")
    verdict = "FEASIBLE (PASS)" if results["overall_fit"] else "EXCEEDS BUDGET (OPTIMIZATION NEEDED)"
    print(f"Arduino Uno Q Compatibility:  {verdict}")
    print("================================================================")


if __name__ == "__main__":
    main()
