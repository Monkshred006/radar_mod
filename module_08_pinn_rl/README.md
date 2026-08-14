# Module 8 — Physics-Informed Reinforcement Learning (PINN + RL)

## 1. Overview & Purpose

Module 8 implements the downstream **Physics-Informed Reinforcement Learning (PINN + RL)** system for PhotonShield AI. It bridges perception and physical action by consuming the continuous latent and decision representations produced by upstream modules (Module 4 Mamba-Hybrid and Module 7 Decision Layer) to construct causal RL states, learn physics-informed transition dynamics, and train control policies.

```
             MODULE 4
          Mamba-Hybrid
                │
                ▼
             MODULE 7
       task probabilities
                │
                ▼
          RL State Builder
                │
                ▼
             RL Policy
                │
              Action
                │
                ▼
        PINN Dynamics Model
                │
          Next State
                │
                ▼
             Reward
                │
                └──────────► RL (PPO)

        Physics residual
                │
                ▼
             PINN loss
```

---

## 2. Distinction of Roles

- **PINN**: Learns a physics-informed transition dynamics model $f_\theta(s_t, a_t) \to \hat{s}_{t+1}$ by minimizing prediction error plus a physics residual penalty ($L_{\text{total}} = L_{\text{data}} + \lambda_{\text{phys}} L_{\text{physics}}$).
- **RL (Policy)**: Learns a policy $\pi_\theta(a_t | s_t)$ mapping states to actions to maximize cumulative reward $R_t$.
- **Environment**: Defines the ground-truth transition dynamics and task specification.

---

## 3. RL State Construction & Dimensionality

The RL state preserves continuous probability and uncertainty information rather than thresholded binary decisions:

$$s_t = [z_{\text{mamba}}, p_{\text{target}}, p_{\text{anomaly}}, e_{\text{env}}, x_{\text{phys}}]$$

where:
- $z_{\text{mamba}}$: Module 4 pooled latent representation (`d_model`, default 128)
- $p_{\text{target}}$: Module 7 continuous target probability
- $p_{\text{anomaly}}$: Module 7 continuous anomaly probability
- $e_{\text{env}}$: Module 7 continuous environmental assessment (`environment_dim`, default 3)
- $x_{\text{phys}}$: Optional physical state variables (application-dependent)

`state_dim` is dynamically derived at runtime from `RLStateConfig.state_dim` and is never hard-coded.

---

## 4. Action Space

Action abstractions supported:
- **Discrete Actions**: e.g., $\{0, 1, 2, 3\}$. Default development labels (`maintain`, `inspect`, `alert`, `reposition`) are **DEVELOPMENT / SYNTHETIC PLACEHOLDERS** and do not represent final control semantics.
- **Continuous Actions**: $a \in \mathbb{R}^n$.

---

## 5. Physics Models & Residuals

1. **`KinematicPhysicsModel` (Default — SYNTHETIC VERIFICATION ONLY)**:
   $$x_{t+1} = x_t + v_t \Delta t, \quad v_{t+1} = v_t + a_t \Delta t$$
   Used strictly for software validation of the PINN + RL pipeline. Does **not** represent real PhotonShield hardware dynamics.

2. **`WaveConvectionPhysicsModel` (Optional — Disabled by Default)**:
   $$R_{\text{wave}} = \frac{\partial^2 u}{\partial t^2} + v \frac{\partial u}{\partial x} - \kappa \frac{\partial^2 u}{\partial x^2} = 0$$
   Adapted from the PhotonPINN-Radar reference paper. Applied only when the state contains a spatiotemporal field dimensionally consistent with $(u, x, t, v, \kappa)$.

3. **`NoPhysicsModel`**: Returns zero residual (used for data-only dynamics, $\lambda_{\text{phys}} = 0$).

---

## 6. PINN Loss vs RL Reward

- **PINN Loss**: $L_{\text{total}} = L_{\text{data}} + \lambda_{\text{phys}} L_{\text{physics}}$ (trains the dynamics model).
- **RL Reward**: $R_t = w_{\text{task}} r_{\text{task}} - w_{\text{err}} e_{\text{state}} - w_{\text{phys}} \text{viol}_{\text{phys}} - w_{\text{act}} c_{\text{act}}$ (trains the policy).

PINN optimization and RL optimization are executed as **separate training processes**. The PINN physics loss does not automatically backpropagate through the RL policy.

---

## 7. Experimental Ablation Matrix

Module 8 supports three distinct experiment configurations:
1. **Experiment A (RL-only)**: No PINN dynamics model; RL interacts with ordinary environment dynamics.
2. **Experiment B (Data-only dynamics)**: Dynamics model trained with $\lambda_{\text{phys}} = 0$ (no physics residual).
3. **Experiment C (RL + PINN)**: Dynamics model trained with $\lambda_{\text{phys}} > 0$; RL interacts with/evaluates against PINN dynamics.

---

## 8. Relation to "PhotonPINN-Radar" Research Paper

The uploaded research paper (*"PhotonPINN-Radar: Physics-Informed Diffusion and Tracking for Photonic FMCW Radar"*) serves as a **design reference**.

- **Source-derived**: The wave-convection PDE residual formulation and physics-informed spatiotemporal regularization.
- **PhotonShield-specific**: Module 4/7 state extraction, PPO RL integration, and the three-way baseline comparison architecture.
- **Limitations**: The reference paper's evaluation is synthetic/simulation-based. No simulated results are claimed as validated real hardware performance.

---

## 9. Staged Training Workflow

- **Stage 1**: Construct synthetic/available transition dataset.
- **Stage 2**: Train PINN dynamics model ($L_{\text{data}} + \lambda_{\text{phys}} L_{\text{physics}}$).
- **Stage 3**: Validate PINN dynamics independently.
- **Stage 4**: Train RL-only baseline (Exp A).
- **Stage 5**: Train RL + PINN (Exp C).
- **Stage 6**: Evaluate all three configurations (A vs B vs C).
