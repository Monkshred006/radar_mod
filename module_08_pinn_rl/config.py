"""Module 8 Configuration — Physics-Informed Reinforcement Learning (PINN + RL).

Separate configuration dataclasses for each concern:
  RLStateConfig    — which Module 4/7 outputs enter the RL state
  PhysicsConfig    — physics residual selection and parameters
  DynamicsConfig   — PINN/dynamics model architecture and training
  RLConfig         — RL algorithm and policy
  EnvironmentConfig — RL environment
  RewardConfig     — reward shaping weights
  PINNRLConfig     — root configuration aggregating all sub-configs

Design notes
------------
* state_dim is always DERIVED from RLStateConfig fields at runtime.
  It is never hard-coded anywhere in Module 8.
* Three distinct experiments are supported (see experiment.py):
    Exp A: RL-only          — no PINN, synthetic environment dynamics
    Exp B: Data-only PINN   — PINN with lambda_phys = 0
    Exp C: RL + PINN        — PINN with lambda_phys > 0
* PINN training and RL training are separate optimization loops.
  The PINN physics loss does NOT backpropagate through the RL policy
  by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# RL State Configuration
# ---------------------------------------------------------------------------

@dataclass
class RLStateConfig:
    """Configures which Module 4/7 outputs are included in the RL state.

    The state preserves continuous uncertainty information rather than
    thresholded binary decisions wherever possible.

    state_dim is computed from enabled components at runtime via the
    `state_dim` property — it is never hard-coded.
    """

    # Module 4 latent representation
    use_mamba_latent: bool = True
    mamba_latent_dim: int = 128          # Must match Module 4 d_model

    # Module 7 continuous probabilities (NOT binary decisions)
    use_target_probability: bool = True
    use_anomaly_probability: bool = True

    # Module 7 environmental outputs
    use_environment: bool = True
    environment_dim: int = 3             # Must match Module 7 num_environment_outputs

    # Optional: raw physical state (application-dependent; disabled by default)
    use_raw_physical_state: bool = False
    physical_state_dim: int = 0

    @property
    def state_dim(self) -> int:
        """Compute state dimension from enabled components. Never hard-coded."""
        dim = 0
        if self.use_mamba_latent:
            dim += self.mamba_latent_dim
        if self.use_target_probability:
            dim += 1
        if self.use_anomaly_probability:
            dim += 1
        if self.use_environment:
            dim += self.environment_dim
        if self.use_raw_physical_state:
            dim += self.physical_state_dim
        return dim

    def validate(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("RLStateConfig: at least one state component must be enabled.")
        if self.use_mamba_latent and self.mamba_latent_dim <= 0:
            raise ValueError("mamba_latent_dim must be > 0 when use_mamba_latent=True.")
        if self.use_environment and self.environment_dim <= 0:
            raise ValueError("environment_dim must be > 0 when use_environment=True.")
        if self.use_raw_physical_state and self.physical_state_dim <= 0:
            raise ValueError(
                "physical_state_dim must be > 0 when use_raw_physical_state=True."
            )


# ---------------------------------------------------------------------------
# Physics Configuration
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    """Selects and parameterises the physics residual.

    Supported physics models
    ------------------------
    "kinematic"
        Discrete kinematic constraint (SYNTHETIC VERIFICATION ONLY):
            x_{t+1} = x_t + v_t * dt
            v_{t+1} = v_t + a_t * dt
        Residual computed via arithmetic — no autograd required.
        Does NOT represent validated PhotonShield hardware dynamics.

    "wave_convection"
        Wave-convection PDE residual (optional; DISABLED BY DEFAULT):
            R = d²u/dt² + v·du/dx - κ·d²u/dx²
        Adapted from the PhotonPINN-Radar design reference paper.
        MUST ONLY be enabled when the state contains a spatiotemporal
        field with physically meaningful (u, x, t, v, κ). NOT applied
        to the general RL state vector.

    "none"
        No physics regularisation. Equivalent to lambda_phys = 0 for
        the data-only PINN baseline (Experiment B).

    lambda_phys
    -----------
    Default: 0.1 — a development default, NOT a validated optimal value.
    Ablation values: {0.0, 0.01, 0.1, 1.0}.
    lambda_phys = 0.0 gives data-only dynamics (Experiment B).
    lambda_phys > 0  gives physics-informed dynamics (Experiment C).
    """

    physics_model: Literal["kinematic", "wave_convection", "none"] = "kinematic"
    lambda_physics: float = 0.1          # Experiment parameter — not validated optimal
    lambda_laplacian: float = 0.0        # Optional Laplacian smoothing (paper: λ_lap)
    dt: float = 0.1                      # Timestep for kinematic residual

    # Kinematic model: state-vector indices for position and velocity components
    position_indices: List[int] = field(default_factory=lambda: [0])
    velocity_indices: List[int] = field(default_factory=lambda: [1])

    # Wave-convection parameters (PhotonPINN-Radar style; see physics.py)
    wave_velocity: float = 1.0           # v: advection velocity
    wave_diffusion: float = 0.01         # κ: diffusion coefficient

    def validate(self) -> None:
        if self.lambda_physics < 0:
            raise ValueError("lambda_physics must be >= 0.")
        if self.dt <= 0:
            raise ValueError("dt must be > 0.")
        if self.wave_diffusion < 0:
            raise ValueError("wave_diffusion (κ) must be >= 0.")


# ---------------------------------------------------------------------------
# Dynamics / PINN Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class DynamicsConfig:
    """Configures the PINN dynamics model  f_θ(s_t, a_t) → ŝ_{t+1}.

    PINN role: learned physics-informed dynamics model.
    Training loss:
        L_total = L_data + lambda_phys * L_physics

    Experiments
    -----------
    Exp B (data-only): lambda_phys = 0  → L_total = L_data only
    Exp C (RL+PINN):   lambda_phys > 0  → physics-informed training

    Note: state_dim must match RLStateConfig.state_dim (or environment
    state_dim for synthetic environments).
    """

    state_dim: int = 133                 # Must match RLStateConfig.state_dim
    action_type: Literal["discrete", "continuous"] = "discrete"
    action_dim: int = 4                  # Discrete: num actions; continuous: R^n
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: Literal["relu", "tanh", "gelu"] = "relu"
    data_loss: Literal["mse", "l1", "smooth_l1"] = "mse"
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 10

    def validate(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be > 0.")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0.")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must not be empty.")


# ---------------------------------------------------------------------------
# RL Configuration
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """Configures the RL algorithm and policy network.

    Supported algorithms
    --------------------
    "ppo": Proximal Policy Optimisation (discrete or continuous actions).
           Implemented without Stable-Baselines3 dependency.

    Action semantics
    ----------------
    Discrete action defaults {0,1,2,3} are DEVELOPMENT PLACEHOLDERS only.
    The final physical action space depends on the PhotonShield application.
    """

    algorithm: Literal["ppo"] = "ppo"
    action_type: Literal["discrete", "continuous"] = "discrete"
    action_dim: int = 4                  # Discrete: num classes; continuous: R^n
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: Literal["relu", "tanh"] = "tanh"
    learning_rate: float = 3e-4
    gamma: float = 0.99                  # Discount factor
    gae_lambda: float = 0.95             # GAE-λ
    clip_eps: float = 0.2                # PPO clipping epsilon
    value_coef: float = 0.5             # Value function loss coefficient
    entropy_coef: float = 0.01          # Entropy bonus coefficient
    max_grad_norm: float = 0.5          # Gradient clipping norm
    n_steps: int = 128                   # Rollout steps per update
    n_epochs: int = 4                    # PPO epochs per rollout
    batch_size: int = 64                 # Mini-batch size

    def validate(self) -> None:
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1].")
        if self.clip_eps <= 0:
            raise ValueError("clip_eps must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")


# ---------------------------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    """Configures the RL environment.

    env_type choices
    ----------------
    "synthetic_kinematic"
        2-D state [position, velocity] — SYNTHETIC VERIFICATION ONLY.
        Does NOT represent real PhotonShield hardware dynamics.
        Used to verify that the PINN + RL pipeline runs correctly.

    "photonshield"
        Future real/replay-driven environment using Module 4/7 outputs.
        Requires real or replay data; not yet implemented.
    """

    env_type: Literal["synthetic_kinematic", "photonshield"] = "synthetic_kinematic"
    state_dim: int = 2                   # Synthetic: [x, v]
    action_type: Literal["discrete", "continuous"] = "discrete"
    n_discrete_actions: int = 3          # Synthetic: {-1, 0, +1} acceleration
    max_steps: int = 200
    target_position: float = 1.0
    dt: float = 0.1
    noise_std: float = 0.01
    seed: int = 42

    def validate(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be > 0.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0.")
        if self.dt <= 0:
            raise ValueError("dt must be > 0.")


# ---------------------------------------------------------------------------
# Reward Configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Configures reward function component weights.

    IMPORTANT: These weights are DEVELOPMENT PLACEHOLDERS.
    The final PhotonShield reward must be validated against real
    application objectives before deployment.

    Physics-violation penalty (weight_physics_violation) is SEPARATE
    from PINN L_physics. They are independent quantities serving
    different optimization objectives.
    """

    weight_task_success: float = 1.0
    weight_state_error: float = 0.1
    weight_physics_violation: float = 0.05    # Separate from PINN L_physics
    weight_action_cost: float = 0.01

    def validate(self) -> None:
        for name, val in [
            ("weight_task_success", self.weight_task_success),
            ("weight_state_error", self.weight_state_error),
            ("weight_physics_violation", self.weight_physics_violation),
            ("weight_action_cost", self.weight_action_cost),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be >= 0.")


# ---------------------------------------------------------------------------
# Root Configuration
# ---------------------------------------------------------------------------

@dataclass
class PINNRLConfig:
    """Root configuration for Module 8 — PINN + Reinforcement Learning.

    Aggregates all sub-configurations. See individual dataclasses for details.

    Three supported experiments (see experiment.py and baselines.py):
        Exp A: RL-only        — no PINN, environment supplies dynamics
        Exp B: Data-only PINN — PINN dynamics, physics_config.lambda_physics = 0
        Exp C: RL + PINN      — PINN dynamics, physics_config.lambda_physics > 0
    """

    state_config: RLStateConfig = field(default_factory=RLStateConfig)
    physics_config: PhysicsConfig = field(default_factory=PhysicsConfig)
    dynamics_config: DynamicsConfig = field(default_factory=DynamicsConfig)
    rl_config: RLConfig = field(default_factory=RLConfig)
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    seed: int = 42
    device: str = "cpu"
    experiment_name: str = "module_08_pinn_rl"

    def validate(self) -> None:
        self.state_config.validate()
        self.physics_config.validate()
        self.dynamics_config.validate()
        self.rl_config.validate()
        self.env_config.validate()
        self.reward_config.validate()
