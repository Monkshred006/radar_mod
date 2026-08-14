"""Mamba Temporal Branch Module with Dual-Backend Support.

Supports:
1. `MambaBackend`: Native CUDA `mamba_ssm` package (when available).
2. `FallbackSSMBackend`: Pure PyTorch selective State Space Model (development/CPU fallback).

Guarantees consistent public API regardless of environment.
"""

from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_04_mamba_hybrid.config import MambaHybridConfig

# Check if mamba_ssm is available
NATIVE_MAMBA_AVAILABLE = False
try:
    import mamba_ssm
    NATIVE_MAMBA_AVAILABLE = True
except ImportError:
    NATIVE_MAMBA_AVAILABLE = False


class FallbackSSMBackend(nn.Module):
    """Pure PyTorch Selective State Space Model (SSM) fallback block.

    Implements:
    1. Input projection to 2 * d_inner (split into x branch and z gate branch)
    2. Causal 1D Convolution along temporal dimension T
    3. SiLU activation
    4. Selective parameter projection for B, C, dt
    5. Discretized state-space recurrent recurrence along time T
    6. Gated output projection (z * y) -> d_model
    """

    def __init__(self, config: MambaHybridConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.mamba_state_dim
        self.d_conv = config.mamba_conv_dim
        self.expand = config.mamba_expand
        self.d_inner = int(self.expand * self.d_model)

        # 1. In projection: d_model -> 2 * d_inner
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # 2. Causal 1D Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,  # Padded for causal shift
            bias=True,
        )

        # 3. Selective SSM Parameter Projections
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # State Space Parameters A (log-space initialization) and D (residual skip)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 4. Out projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Initialize dt_proj bias to log(exp(rand) - 1 + 1e-4) for stable SSM time steps
        dt_init_std = self.d_inner**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for FallbackSSMBackend.

        Args:
            x: Input tensor [B, T, D_model].

        Returns:
            Output tensor [B, T, D_model].
        """
        B, T, D = x.shape

        # 1. Project to x_branch and z_gate
        xz = self.in_proj(x)  # [B, T, 2 * d_inner]
        x_branch, z_gate = xz.chunk(2, dim=-1)  # each [B, T, d_inner]

        # 2. Causal 1D Conv along T
        # Conv1d expects [B, C, T]
        x_conv = x_branch.transpose(1, 2)  # [B, d_inner, T]
        x_conv = self.conv1d(x_conv)[:, :, :T]  # Truncate causal padding to length T
        x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]

        # 3. Activation
        x_act = F.silu(x_conv)  # [B, T, d_inner]

        # 4. Selective SSM parameter calculation
        # x_proj outputs B_t (d_state), C_t (d_state), dt_raw (1)
        ssm_params = self.x_proj(x_act)  # [B, T, 2*d_state + 1]
        B_ssm, C_ssm, dt_raw = ssm_params.split([self.d_state, self.d_state, 1], dim=-1)

        # Compute dt per feature channel: [B, T, d_inner]
        dt = F.softplus(self.dt_proj(dt_raw))  # [B, T, d_inner]

        # A matrix: -exp(A_log) [d_inner, d_state]
        A = -torch.exp(self.A_log)

        # 5. Causal Recurrent Selective State-Space Scan along T
        y_list = []
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)  # [B, d_inner, d_state]

        for t in range(T):
            # dt_t: [B, d_inner, 1]
            dt_t = dt[:, t, :].unsqueeze(-1)
            # A_bar = exp(A * dt_t): [B, d_inner, d_state]
            dA = torch.exp(A.unsqueeze(0) * dt_t)

            # B_t: [B, 1, d_state]
            Bt = B_ssm[:, t, :].unsqueeze(1)
            # x_t: [B, d_inner, 1]
            xt = x_act[:, t, :].unsqueeze(-1)

            # dB = dt_t * Bt * xt: [B, d_inner, d_state]
            dB = dt_t * Bt * xt

            # State recurrence: h_t = dA * h_{t-1} + dB
            h = dA * h + dB

            # Output projection: y_t = (h_t @ C_t^T) + D * x_t
            Ct = C_ssm[:, t, :].unsqueeze(-1)  # [B, d_state, 1]
            yt = torch.matmul(h, Ct).squeeze(-1)  # [B, d_inner]
            yt = yt + self.D * x_act[:, t, :]  # Add skip connection D
            y_list.append(yt)

        y = torch.stack(y_list, dim=1)  # [B, T, d_inner]

        # 6. Gated output with z_gate
        out = y * F.silu(z_gate)  # [B, T, d_inner]
        out = self.out_proj(out)  # [B, T, D_model]

        return out


class MambaTemporalBranch(nn.Module):
    """Mamba Temporal SSM Branch with Backend Selection.

    Maintains long-range causal temporal dependencies across time T.
    """

    def __init__(self, config: MambaHybridConfig):
        super().__init__()
        self.config = config

        # Determine backend
        req_backend = config.backend
        if req_backend == "mamba-ssm":
            if not NATIVE_MAMBA_AVAILABLE:
                raise RuntimeError(
                    "Requested backend 'mamba-ssm' but `mamba_ssm` library is not installed."
                )
            self._backend_type = "mamba-ssm"
        elif req_backend == "fallback":
            self._backend_type = "fallback"
        else:  # "auto"
            self._backend_type = "mamba-ssm" if NATIVE_MAMBA_AVAILABLE else "fallback"

        if self._backend_type == "mamba-ssm":
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=config.d_model,
                d_state=config.mamba_state_dim,
                d_conv=config.mamba_conv_dim,
                expand=config.mamba_expand,
            )
        else:
            self.mamba = FallbackSSMBackend(config)

    @property
    def backend_name(self) -> str:
        """Return human-readable backend name."""
        return self._backend_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequence tensor [B, T, D_model].

        Returns:
            Output sequence tensor [B, T, D_model].
        """
        if not self.config.use_mamba:
            return x

        return self.mamba(x)
