"""Module 8: Physics-Informed Reinforcement Learning (PINN + RL) for PhotonShield AI.

Consumes continuous representations from Module 4 (`pooled_output`) and Module 7
(`target_probability`, `anomaly_probability`, `environmental_assessment`) to build
causal, uncertainty-preserving RL states, and trains physics-informed dynamics models
and RL policies.
"""

from module_08_pinn_rl.action import ActionEncoder, ActionSpec
from module_08_pinn_rl.baselines import DataOnlyDynamics, RLOnlyBaseline, RLPINNSystem
from module_08_pinn_rl.checkpointing import (
    load_pinn_checkpoint,
    load_rl_checkpoint,
    save_pinn_checkpoint,
    save_rl_checkpoint,
)
from module_08_pinn_rl.config import (
    DynamicsConfig,
    EnvironmentConfig,
    PhysicsConfig,
    PINNRLConfig,
    RewardConfig,
    RLConfig,
    RLStateConfig,
)
from module_08_pinn_rl.dynamics import PhysicsInformedDynamicsModel
from module_08_pinn_rl.environment import (
    PhotonShieldRLEnv,
    SyntheticKinematicEnv,
    build_environment,
)
from module_08_pinn_rl.evaluation import ComparisonEvaluator, PINNEvaluator, RLEvaluator
from module_08_pinn_rl.experiment import ExperimentRunner
from module_08_pinn_rl.physics import (
    KinematicPhysicsModel,
    NoPhysicsModel,
    PhysicsModel,
    WaveConvectionPhysicsModel,
    build_physics_model,
)
from module_08_pinn_rl.pinn import PINNLoss
from module_08_pinn_rl.profiling import (
    profile_module_08_pipeline,
    profile_pinn_model,
    profile_rl_policy,
)
from module_08_pinn_rl.replay import ReplayBuffer, RolloutBuffer
from module_08_pinn_rl.residuals import (
    compute_kinematic_residual,
    compute_wave_convection_residual,
)
from module_08_pinn_rl.reward import RewardFunction
from module_08_pinn_rl.rl_algorithm import PPO
from module_08_pinn_rl.rl_policy import MLPPolicy
from module_08_pinn_rl.state import RLState, RLStateBuilder
from module_08_pinn_rl.training import PINNTrainer, RLTrainer, StagedTrainer
from module_08_pinn_rl.transitions import Episode, Transition

__all__ = [
    # Configurations
    "RLStateConfig",
    "PhysicsConfig",
    "DynamicsConfig",
    "RLConfig",
    "EnvironmentConfig",
    "RewardConfig",
    "PINNRLConfig",
    # State & Action
    "RLState",
    "RLStateBuilder",
    "ActionSpec",
    "ActionEncoder",
    "Transition",
    "Episode",
    # Physics & Residuals
    "PhysicsModel",
    "KinematicPhysicsModel",
    "WaveConvectionPhysicsModel",
    "NoPhysicsModel",
    "build_physics_model",
    "compute_kinematic_residual",
    "compute_wave_convection_residual",
    # Dynamics & PINN Loss
    "PhysicsInformedDynamicsModel",
    "PINNLoss",
    # RL & Environment
    "RewardFunction",
    "SyntheticKinematicEnv",
    "PhotonShieldRLEnv",
    "build_environment",
    "ReplayBuffer",
    "RolloutBuffer",
    "MLPPolicy",
    "PPO",
    # Training & Evaluation
    "PINNTrainer",
    "RLTrainer",
    "StagedTrainer",
    "PINNEvaluator",
    "RLEvaluator",
    "ComparisonEvaluator",
    "save_pinn_checkpoint",
    "load_pinn_checkpoint",
    "save_rl_checkpoint",
    "load_rl_checkpoint",
    "profile_pinn_model",
    "profile_rl_policy",
    "profile_module_08_pipeline",
    # Baselines & Experiments
    "RLOnlyBaseline",
    "DataOnlyDynamics",
    "RLPINNSystem",
    "ExperimentRunner",
]
