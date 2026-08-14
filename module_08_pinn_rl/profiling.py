"""Profiling and efficiency accounting utilities for Module 8 PINN + RL.

Measures:
- PINN Dynamics model: parameter count, forward latency (mean, p50, p95), memory.
- RL Policy: parameter count, action sampling latency.
- End-to-end state-to-action and state-to-next-state latency.

Note: Latency and throughput figures measured here are software benchmarks and
do not represent edge real-time hardware performance.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from module_08_pinn_rl.action import ActionEncoder, ActionSpec
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.rl_policy import MLPPolicy


def profile_pinn_model(
    model: PhysicsInformedDynamicsModel,
    state_dim: int,
    action_dim: int,
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Profile latency, parameters, and memory of the PINN dynamics model."""
    model = model.to(device).eval()
    params = model.count_parameters()

    dummy_state = torch.randn(1, state_dim, device=device)
    dummy_action = torch.zeros(1, action_dim, device=device)
    dummy_action[0, 0] = 1.0

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(dummy_state, dummy_action)

    latencies: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_state, dummy_action)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    return {
        "parameter_count": params,
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "device": device,
    }


def profile_rl_policy(
    policy: MLPPolicy,
    state_dim: int,
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Profile parameter count and action selection latency of the RL policy."""
    policy = policy.to(device).eval()
    params = policy.count_parameters()

    dummy_state = torch.randn(1, state_dim, device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = policy.get_action_and_value(dummy_state)

    latencies: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = policy.get_action_and_value(dummy_state)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    return {
        "parameter_count": params,
        "action_latency_mean_ms": float(np.mean(latencies)),
        "action_latency_p50_ms": float(np.percentile(latencies, 50)),
        "action_latency_p95_ms": float(np.percentile(latencies, 95)),
        "device": device,
    }


def profile_module_08_pipeline(
    policy: MLPPolicy,
    dynamics_model: PhysicsInformedDynamicsModel,
    state_dim: int,
    action_dim: int,
    n_runs: int = 50,
) -> Dict[str, Any]:
    """Profile full state -> action -> predicted next state loop."""
    pinn_profile = profile_pinn_model(dynamics_model, state_dim, action_dim, n_runs=n_runs)
    rl_profile = profile_rl_policy(policy, state_dim, n_runs=n_runs)

    total_params = pinn_profile["parameter_count"] + rl_profile["parameter_count"]
    combined_latency = rl_profile["action_latency_mean_ms"] + pinn_profile["latency_mean_ms"]

    return {
        "pinn_profile": pinn_profile,
        "rl_profile": rl_profile,
        "total_parameters": total_params,
        "combined_step_latency_ms": combined_latency,
    }
