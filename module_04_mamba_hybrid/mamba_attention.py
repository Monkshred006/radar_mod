"""Hybrid Mamba + Attention Layers for Future V1/V2 Staged Scaling.

Provides:
- `MambaAttentionHybridBlock`: Integrates a Mamba SSM branch and an optional self-attention branch with configurable fusion gating.
- `OptionalAttentionLayer`: Light Multi-Head Self-Attention block with gating.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_04_mamba_hybrid.mamba_core import MiniMambaBlock


class OptionalAttentionLayer(nn.Module):
    """Lightweight Multi-Head Self-Attention with residual connection."""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor `[B, T, d_model]`.
            key_padding_mask: Optional mask `[B, T]`.
            attn_mask: Optional attention mask `[T, T]`.

        Returns:
            Output tensor `[B, T, d_model]`.
        """
        residual = x
        x_norm = self.norm(x)
        attn_out, _ = self.mha(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return residual + self.dropout(attn_out)


class MambaAttentionHybridBlock(nn.Module):
    """Hybrid block cascading Mini-Mamba and optional Multi-Head Attention.

    V0 executes strictly Mamba without cross-attention.
    V1/V2 can enable attention branch via `use_attention=True`.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.mamba_block = MiniMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            backend=backend,
        )

        if self.use_attention:
            self.attn_block = OptionalAttentionLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.attn_block = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor `[B, T, d_model]`.
            mask: Optional mask.

        Returns:
            Output tensor `[B, T, d_model]`.
        """
        out = self.mamba_block(x)
        if self.use_attention and self.attn_block is not None:
            out = self.attn_block(out, key_padding_mask=mask)
        return out
