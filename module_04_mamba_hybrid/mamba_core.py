"""Core Mamba / State-Space Model (SSM) Blocks for PhotonShield AI.

Provides:
- `MambaCoreLayer`: Standalone single-layer selective SSM block with dual backend (CUDA `mamba_ssm` or pure PyTorch `FallbackSSMBackend`).
- `MiniMambaBlock`: Lightweight residual Mamba block designed for embedded / micro-edge perception (e.g. Arduino Uno Q target).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check native mamba_ssm availability
NATIVE_MAMBA_AVAILABLE = False
try:
    import mamba_ssm
    NATIVE_MAMBA_AVAILABLE = True
except ImportError:
    NATIVE_MAMBA_AVAILABLE = False


class PurePyTorchSSM(nn.Module):
    """Pure PyTorch Selective State Space Model implementation.

    Computes:
    1. Input projection: x -> [x_branch, z_gate] (dim = 2 * d_inner)
    2. Causal 1D convolution over temporal dimension T
    3. SiLU activation
    4. Input-dependent SSM parameter projections: B_t, C_t, dt_t
    5. Discretized state space recurrence: h_t = dA * h_{t-1} + dB * x_t, y_t = h_t @ C_t^T + D * x_t
    6. Output gating: y * SiLU(z_gate) -> out_proj -> d_model
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 1. In-projection: d_model -> 2 * d_inner
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # 2. Causal 1D Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )

        # 3. Parameter projections: B, C, dt
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # SSM Continuous parameters: A (log-space), D (skip connection)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 4. Out-projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        dt_init_std = self.d_inner**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for PurePyTorchSSM.

        Args:
            x: Input tensor of shape `[B, T, D]` where D == d_model.

        Returns:
            Output tensor of shape `[B, T, D]`.
        """
        B, T, D = x.shape

        # 1. Project input
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)

        # 2. Causal 1D Conv
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = x_conv.transpose(1, 2)

        # 3. Activation
        x_act = F.silu(x_conv)

        # 4. Selective SSM params
        ssm_params = self.x_proj(x_act)
        B_ssm, C_ssm, dt_raw = ssm_params.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))

        A = -torch.exp(self.A_log)

        # 5. Causal Recurrent Scan
        y_list = []
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)  # [B, d_inner, 1]
            dA = torch.exp(A.unsqueeze(0) * dt_t)  # [B, d_inner, d_state]

            Bt = B_ssm[:, t, :].unsqueeze(1)  # [B, 1, d_state]
            xt = x_act[:, t, :].unsqueeze(-1)  # [B, d_inner, 1]

            dB = dt_t * Bt * xt  # [B, d_inner, d_state]
            h = dA * h + dB

            Ct = C_ssm[:, t, :].unsqueeze(-1)  # [B, d_state, 1]
            yt = torch.matmul(h, Ct).squeeze(-1)  # [B, d_inner]
            yt = yt + self.D * x_act[:, t, :]
            y_list.append(yt)

        y = torch.stack(y_list, dim=1)  # [B, T, d_inner]

        # 6. Gated output
        out = y * F.silu(z_gate)
        return self.out_proj(out)


class MiniMambaBlock(nn.Module):
    """Lightweight Mini-Mamba Block with pre-LayerNorm and residual connection.

    Architecture:
        x -> LayerNorm -> Mamba SSM Core -> Dropout -> (+) -> Output
        |______________________________________________|
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        # Select backend
        if backend == "mamba-ssm" or (backend == "auto" and NATIVE_MAMBA_AVAILABLE):
            try:
                from mamba_ssm import Mamba
                self.ssm = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.backend_name = "mamba-ssm"
            except Exception:
                self.ssm = PurePyTorchSSM(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
                )
                self.backend_name = "fallback"
        else:
            self.ssm = PurePyTorchSSM(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.backend_name = "fallback"

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MiniMambaBlock.

        Args:
            x: Input tensor `[B, T, d_model]`.

        Returns:
            Output tensor `[B, T, d_model]`.
        """
        residual = x
        x_norm = self.norm(x)
        out = self.ssm(x_norm)
        out = self.dropout(out)
        return residual + out
