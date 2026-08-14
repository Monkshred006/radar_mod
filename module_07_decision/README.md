# Module 7 — PhotonShield Decision / Task Output Layer

## Overview

Module 7 is the application-level **Decision & Task Output Layer** for PhotonShield AI. It maps the latent representation produced by Module 4 (`pooled_output` `[B, D_model]`) to task-specific prediction heads and deterministic application decision logic.

> [!IMPORTANT]
> **Real Target Label Disclaimer**:
> The documented PhotonShield objectives are Target Indication, Anomaly Detection, and Environmental Assessment. However, the final real-world class taxonomy and ground-truth target schema are not yet finalized. Module 7 provides fully configurable schemas. Synthetic data is used strictly for software verification and does **not** reflect real-world performance metrics.

---

## Conceptual Architecture Flow

```
                  MODULE 4 / MODULE 6
                 Mamba-Hybrid Engine
                          │
                          ▼
                    pooled_output [B, D_model]
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
  TargetHead         AnomalyHead     EnvironmentalHead
 (Classification)     (Binary)      (Reg/Classification)
       │                  │                  │
       ▼                  ▼                  ▼
  target_logits      anomaly_logits  environment_output
   [B, N_target]         [B, 1]          [B, N_env]
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                Model Output Dictionary
                          │
                          ▼
               Temperature Calibration (T)
                          │
                          ▼
                    DecisionLogic
             (Softmax, Sigmoid, Thresholds,
             Causal Decision Smoothing, Hysteresis)
                          │
                          ▼
            PhotonShield Application Output
```

---

## Task Heads

1. **Target Indication Head (`TargetHead`)**:
   - Binary or multi-class target classification.
   - Outputs unnormalized logits `[B, num_target_classes]`. Does NOT apply argmax inside the neural head.
2. **Anomaly Detection Head (`AnomalyHead`)**:
   - Binary anomaly detection.
   - Outputs logits `[B, 1]`. Uses `BCEWithLogitsLoss` with optional positive class weighting.
3. **Environmental Assessment Head (`EnvironmentalHead`)**:
   - **Regression mode**: Outputs continuous values `[B, num_environment_outputs]` (e.g., temperature, humidity, pressure).
   - **Classification mode**: Outputs environmental category logits `[B, num_environment_classes]`.

---

## Output vs Decision Separation

- **Model Outputs**: Unnormalized neural logits, continuous predictions, and calibrated probabilities produced by `PhotonShieldMultiTask` and `DecisionLogic`.
- **Application Decisions**: Computed deterministically by `DecisionLogic`:
  - Applies temperature calibration ($z / T$).
  - Probability conversion (`softmax`, `sigmoid`).
  - Dual-threshold hysteresis ($T_{\text{on}}$, $T_{\text{off}}$).
  - Causal decision smoothing (EMA, moving majority vote, minimum consecutive detections).
  - Combined event state output (`NORMAL`, `TARGET`, `ANOMALY`, `TARGET_AND_ANOMALY`).

---

## Downstream Interface for Future PINN + RL Module

Module 7 provides both continuous model representations and application-level decisions on the `PhotonShieldDecisionOutput` structure:

1. **Mamba Latent Representation**: `output.pooled_output` (`[D_model]` tensor reference).
2. **Target Logits & Probabilities**: `output.target_logits`, `output.target_probability`, `output.target_probabilities`.
3. **Anomaly Logits & Probabilities**: `output.anomaly_logits`, `output.anomaly_probability`.
4. **Environmental Outputs**: `output.environmental_assessment`.
5. **Application-Level Decisions**: `output.target_detected`, `output.anomaly_detected`, `output.combined_event_state`.

Future Module 8 may consume these continuous fields to construct its Reinforcement Learning (RL) state space.

> [!NOTE]
> **Important Boundary Note**:
> PINN (Physics-Informed Neural Networks) and RL (Reinforcement Learning) are **NOT** implemented in Module 7. They are explicitly reserved for future Module 8.

```
                     MODULE 4
                  Mamba-Hybrid
                       │
                       ▼
                 pooled_output
                       │
                       ▼
                  MODULE 7
                       │
          ┌────────────┼─────────────┐
          │            │             │
          ▼            ▼             ▼
        Target       Anomaly     Environment
      probability  probability    output
          │            │             │
          └────────────┼─────────────┘
                       │
                       ▼
                  Decisions

             ┌───────────────────┐
             │  Future Module 8  │ (NOT in Module 7)
             └─────────┬─────────┘
                       ▲
                       │
          pooled_output + continuous
          task information
                       │
                       ▼
                  RL state
                       │
                       ▼
                      RL
                       │
                     Action
                       │
                       ▼
                     PINN
```

---

## Streaming Causality Guarantee

All decision smoothing filters operate strictly causally:
$$\text{decision}(t) = f(p_0, p_1, \dots, p_t)$$

Future predictions ($p_{t+1}, p_{t+2}, \dots$) have zero influence on decision state or continuous outputs at time $t$.

---

## Model Precision Compatibility

Module 7 task heads operate downstream of both:
- **FP32 Module 4** Mamba-Hybrid engine.
- **BitNet-Style Module 4** ternary quantized engine (`module_06_bitnet`).
