"""Checkpointing utilities for Module 8 PINN + RL.

Saves and loads:
- PINN / Dynamics model checkpoints (weights, optimizer state, configs, history)
- RL Policy checkpoints (weights, optimizer state, configs, history)

Maintains full separation between PINN and RL policy models. Never overwrites
checkpoints from Modules 4–7.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from module_08_pinn_rl.config import DynamicsConfig, PINNRLConfig, PhysicsConfig, RLConfig
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.rl_policy import MLPPolicy


def save_pinn_checkpoint(
    dynamics_model: PhysicsInformedDynamicsModel,
    path: Union[str, Path],
    config: Optional[PINNRLConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    training_history: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a PINN / Dynamics model checkpoint."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        "model_state": dynamics_model.state_dict(),
        "dynamics_config": asdict(dynamics_model.config),
        "module": "module_08_pinn_rl",
        "type": "pinn_dynamics",
    }
    if config is not None:
        state_dict["full_config"] = asdict(config)
        state_dict["physics_config"] = asdict(config.physics_config)
    if optimizer is not None:
        state_dict["optimizer_state"] = optimizer.state_dict()
    if training_history is not None:
        state_dict["training_history"] = training_history
    if metadata is not None:
        state_dict["metadata"] = metadata

    torch.save(state_dict, save_path)
    return save_path


def load_pinn_checkpoint(
    path: Union[str, Path],
    device: str = "cpu",
) -> Tuple[PhysicsInformedDynamicsModel, Dict[str, Any]]:
    """Load a PINN / Dynamics model checkpoint."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"PINN checkpoint not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device)
    dyn_cfg_dict = checkpoint["dynamics_config"]
    dyn_cfg = DynamicsConfig(**dyn_cfg_dict)

    model = PhysicsInformedDynamicsModel(dyn_cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    return model, checkpoint


def save_rl_checkpoint(
    policy: MLPPolicy,
    path: Union[str, Path],
    state_dim: int,
    config: Optional[PINNRLConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    training_history: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save an RL policy checkpoint."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        "policy_state": policy.state_dict(),
        "state_dim": state_dim,
        "rl_config": asdict(policy.config),
        "module": "module_08_pinn_rl",
        "type": "rl_policy",
    }
    if config is not None:
        state_dict["full_config"] = asdict(config)
        state_dict["reward_config"] = asdict(config.reward_config)
    if optimizer is not None:
        state_dict["optimizer_state"] = optimizer.state_dict()
    if training_history is not None:
        state_dict["training_history"] = training_history
    if metadata is not None:
        state_dict["metadata"] = metadata

    torch.save(state_dict, save_path)
    return save_path


def load_rl_checkpoint(
    path: Union[str, Path],
    device: str = "cpu",
) -> Tuple[MLPPolicy, Dict[str, Any]]:
    """Load an RL policy checkpoint."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"RL checkpoint not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device)
    rl_cfg_dict = checkpoint["rl_config"]
    rl_cfg = RLConfig(**rl_cfg_dict)
    state_dim = checkpoint["state_dim"]

    policy = MLPPolicy(state_dim, rl_cfg)
    policy.load_state_dict(checkpoint["policy_state"])
    policy.to(device)

    return policy, checkpoint
