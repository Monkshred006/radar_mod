"""PhotonShield AI - Phase V0 Perception Model (PhotonV0).

Minimal, hardware-aware temporal perception model optimized for future Arduino Uno Q deployment.

Architecture Pipeline:
    Input [B, T, D]
        ↓
    Linear / Conv1D Embedding (D -> hidden_dim=64)
        ↓
    Mini-Mamba Block 1 (Selective SSM + residual + LayerNorm)
        ↓
    Mini-Mamba Block 2 (Selective SSM + residual + LayerNorm)
        ↓
    LayerNorm
        ↓
    Three Task Heads:
        - detection_head      -> [B, 1] (Target presence probability / confidence)
        - classification_head -> [B, num_classes] (Target class logits)
        - anomaly_head        -> [B, 1] (Continuous anomaly / uncertainty score)
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_04_mamba_hybrid.mamba_core import MiniMambaBlock, NATIVE_MAMBA_AVAILABLE
from module_04_mamba_hybrid.mamba_attention import MambaAttentionHybridBlock


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Calculate the number of parameters in a PyTorch module.

    Args:
        model: PyTorch nn.Module.
        trainable_only: If True, only count parameters requiring gradient.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


class DetectionHead(nn.Module):
    """Detection head predicting target presence probability [B, 1]."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Pooled latent tensor `[B, hidden_dim]`.

        Returns:
            Detection confidence tensor `[B, 1]`.
        """
        return self.net(x)


class ClassificationHead(nn.Module):
    """Classification head predicting target class logits [B, num_classes]."""

    def __init__(self, hidden_dim: int = 64, num_classes: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Pooled latent tensor `[B, hidden_dim]`.

        Returns:
            Class logits tensor `[B, num_classes]`.
        """
        return self.net(x)


class AnomalyHead(nn.Module):
    """Anomaly head predicting continuous anomaly / uncertainty score [B, 1]."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Pooled latent tensor `[B, hidden_dim]`.

        Returns:
            Anomaly score tensor `[B, 1]`.
        """
        return self.net(x)


class PhotonV0(nn.Module):
    """PhotonShield AI Phase V0 Minimal Perception Stack.

    Attributes:
        input_dim: Input feature dimension per timestep (default 64).
        hidden_dim: Model hidden dimension (default 64).
        num_layers: Number of stacked Mini-Mamba blocks (default 2).
        sequence_length: Default / expected sequence length T (default 16).
        num_classes: Number of classification classes (default 4).
        use_attention: False for V0 minimal path.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        sequence_length: int = 16,
        num_classes: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        use_attention: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.use_attention = use_attention

        # 1. Input Linear Projection / Embedding
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # 2. Sequential Mini-Mamba Blocks
        self.layers = nn.ModuleList([
            MambaAttentionHybridBlock(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_attention=use_attention,
                dropout=dropout,
                backend=backend,
            )
            for _ in range(num_layers)
        ])

        # 3. Final LayerNorm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # 4. Multi-Task Perception Heads
        self.detection_head = DetectionHead(hidden_dim=hidden_dim, dropout=dropout)
        self.classification_head = ClassificationHead(
            hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout
        )
        self.anomaly_head = AnomalyHead(hidden_dim=hidden_dim, dropout=dropout)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total parameter count."""
        return count_parameters(self, trainable_only=trainable_only)

    def extract_latents(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract sequence latent state and pooled latent representation.

        Args:
            x: Input tensor `[B, T, D]`.

        Returns:
            Tuple of:
                - latent_seq: `[B, T, H]`
                - pooled_latent: `[B, H]` (last timestep / causal representation)
        """
        # Shape check & projection
        h = self.input_proj(x)  # [B, T, H]

        # Pass through Mini-Mamba blocks
        for layer in self.layers:
            h = layer(h)

        latent_seq = self.final_norm(h)  # [B, T, H]
        pooled_latent = latent_seq[:, -1, :]  # Causal last-step pooling [B, H]
        return latent_seq, pooled_latent

    def forward(
        self,
        x: torch.Tensor,
        return_latents: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for PhotonV0.

        Args:
            x: Input tensor `[B, T, D]`.
            return_latents: If True, includes 'latent' and 'pooled_latent' in output dict.

        Returns:
            Dict containing:
                - 'detection': [B, 1]
                - 'classification': [B, num_classes]
                - 'anomaly': [B, 1]
                - (optional) 'latent': [B, T, hidden_dim]
                - (optional) 'pooled_latent': [B, hidden_dim]
        """
        latent_seq, pooled_latent = self.extract_latents(x)

        det_out = self.detection_head(pooled_latent)
        cls_out = self.classification_head(pooled_latent)
        ano_out = self.anomaly_head(pooled_latent)

        outputs = {
            "detection": det_out,
            "classification": cls_out,
            "anomaly": ano_out,
        }

        if return_latents:
            outputs["latent"] = latent_seq
            outputs["pooled_latent"] = pooled_latent

        return outputs
