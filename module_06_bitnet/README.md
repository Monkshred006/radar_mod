# Module 6 — BitNet / 1.58-Bit Model Optimization & Benchmarking

## Overview

Module 6 investigates whether the PhotonShield Mamba-Hybrid engine can be adapted to a BitNet-style low-bit architecture using approximately 1.58-bit ternary weights:
$$W \in \{-1, 0, +1\}$$

> [!WARNING]
> **Important Hardware & Storage Disclaimer**:
> `1.58-bit` refers to the theoretical information content of ternary weights ($\log_2(3) \approx 1.585$ bits per symbol). It does **not** by itself guarantee 1.58-bit physical file storage, runtime tensor storage, 1.58-bit hardware arithmetic, or execution speedup. Dedicated hardware kernels are required for native ternary acceleration.

---

## Conceptual Architecture Flow

```
                    FP32 CHECKPOINT (Module 5)
                           │
                           ▼
                    Layer Inspection
                           │
                           ▼
                  Selective Replacement
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
        BitLinear                    FP32 layers
     (Ternary Weights)           (Mamba Core, LN)
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  BitNet-Compatible Model
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
      Direct-Ternary PTQ        BitNet-Style QAT
       (No fine-tuning)        (STE Fine-tuning)
               │                       │
               └───────────┬───────────┘
                           ▼
                 Module 5 Evaluator API
               (Identical test dataset split)
                           │
                           ▼
               FP32 / PTQ / QAT Comparison Matrix
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
         Accuracy        Memory         Latency
```

---

## Mathematical Formulation & Autograd

### 1. Scaling Strategy
Weight scaling factor $S$:
$$S = \text{compute\_weight\_scale}(W)$$

- **`mean_abs`** (default): $S = \frac{1}{N} \sum_{i,j} |W_{ij}|$ (mean-absolute-weight scaling).
- **`max_abs`**: $S = \max_{i,j} |W_{ij}|$.
- Scope: `per_tensor` (scalar) or `per_channel` (vector along output dimension).

### 2. Ternary Quantization
Discrete ternary symbols $W_{\text{ternary}} \in \{-1, 0, +1\}$:
$$W_{\text{ternary}} = \text{clip}\left(\text{round}\left(\frac{W}{S}\right), -1, +1\right)$$

Scaled weight used in forward computation:
$$W_{\text{quant}} = S \cdot W_{\text{ternary}}$$

### 3. Straight-Through Estimator (STE)
Because `round()` is non-differentiable (derivative is 0 almost everywhere), QAT uses an STE surrogate gradient:
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W_{\text{quant}}}$$

FP32 master weights $W$ remain trainable and receive updates during backpropagation. The forward pass uses $W_{\text{quant}}$.

---

## Selective Layer Quantization Policy

| Layer Component | Quantization Status | Precision | Rationale |
|---|---|---|---|
| Input Projection | **Ternary** | `BitLinear` | Primary feature projection |
| Sensor Attention Q/K/V/O | **Ternary** | `BitLinear` | Cross-sensor interaction weights |
| FFN Projections | **Ternary** | `BitLinear` | Largest parameter block |
| Mamba Recurrent Core | **FP32** | `nn.Linear` / custom | Sensitive continuous state-space dynamics |
| LayerNorm | **FP32** | `nn.LayerNorm` | Critical for numerical stability |
| Task Head | **FP32** | `nn.Linear` | Output linear projection |

> **Mamba Core Precision Note**: Selected linear layers are ternarized while the Mamba internal state-space operations remain at higher precision.

---

## Experiment Matrix Labels

- **`FP32 Baseline`**: Reference unquantized model trained in Module 5.
- **`Direct-Ternary PTQ`**: Direct weight ternarization of FP32 weights without fine-tuning.
- **`BitNet-Style QAT`**: Quantization-Aware Training initializing from FP32 weights and fine-tuning with STE.

---

## CLI Reference

```bash
# Model Conversion
python -m module_06_bitnet.convert --checkpoint checkpoints/fp32/model.pt --output checkpoints/bitnet/converted.pt

# QAT Fine-Tuning
python -m module_06_bitnet.train --checkpoint checkpoints/bitnet/converted.pt --epochs 5

# Evaluation
python -m module_06_bitnet.evaluate --checkpoint checkpoints/bitnet/bitnet_qat.pt

# Comparison Matrix Generation
python -m module_06_bitnet.compare --fp32 checkpoints/fp32/model.pt --output reports/bitnet

# Profiling
python -m module_06_bitnet.profile --bitnet checkpoints/bitnet/bitnet_qat.pt
```
