"""Reproducibility utilities for Module 5.

Sets deterministic seeds across Python, NumPy, and PyTorch.
Documents unavoidable sources of non-determinism.
"""

from __future__ import annotations
import random
import os
from typing import Dict, Any, Optional
import numpy as np
import torch


def set_seed(seed: int, deterministic_cuda: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic_cuda: If True, enable deterministic CUDA ops
            (may significantly reduce GPU throughput).

    Unavoidable sources of non-determinism when num_workers > 0:
        - DataLoader worker processes receive derived seeds from the main seed,
          but their exact interleaving depends on OS scheduling.
        - Some PyTorch CUDA operations (scatter, gather on certain GPUs) may
          not have a fully deterministic implementation even with this flag.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed_state() -> Dict[str, Any]:
    """Capture current RNG state for checkpoint-level continuity.

    Returns:
        Dict containing Python, NumPy, and PyTorch RNG states.
    """
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_seed_state(state: Dict[str, Any]) -> None:
    """Restore RNG state captured by get_seed_state().

    Args:
        state: Dict from get_seed_state().
    """
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])
