"""Core PhotonMambaHybrid Engine Module for Module 4.

Main neural network engine modeling long-range temporal dependencies and cross-sensor
interactions to produce task-agnostic latent representations.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn

from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.input_projection import SensorTokenProjection
from module_04_mamba_hybrid.temporal_encoding import TemporalEncoding
from module_04_mamba_hybrid.hybrid_block import HybridBlock
from module_04_mamba_hybrid.normalization import get_normalization_layer
from module_04_mamba_hybrid.pooling import SequencePooling

EngineOutput = Dict[str, Any]


class PhotonMambaHybrid(nn.Module):
    """PhotonShield Mamba-Hybrid Core Engine.

    Architecture:
    Module 3 Dict -> Input Projection -> Temporal Encoding -> Stack of N HybridBlocks -> Final Norm -> Sequence Pooling -> EngineOutput Dict.
    """

    def __init__(self, config: Optional[MambaHybridConfig] = None):
        super().__init__()
        self.config = config or MambaHybridConfig()
        self.config.validate()

        # 1. Input Projection
        self.input_projection = SensorTokenProjection(self.config)

        # 2. Temporal / Positional Encoding
        self.temporal_encoding = TemporalEncoding(self.config)

        # 3. Stack of Hybrid Blocks
        self.blocks = nn.ModuleList([
            HybridBlock(self.config) for _ in range(self.config.num_layers)
        ])

        # 4. Final Normalization
        self.final_norm = get_normalization_layer(
            self.config.d_model, self.config.normalization
        )

        # 5. Sequence Pooling
        self.pooling = SequencePooling(self.config)

    @property
    def backend_name(self) -> str:
        """Return backend used by Mamba blocks."""
        if len(self.blocks) > 0 and self.config.use_mamba:
            return self.blocks[0].mamba_branch.backend_name
        return "none"

    def forward(
        self,
        module3_output: Dict[str, Any],
        override_timestamps: Optional[torch.Tensor] = None,
    ) -> EngineOutput:
        """Forward pass consuming Module 3 output dictionary.

        Args:
            module3_output: Dict containing 'tokens', 'token_mask', 'features', 'timestamps'.
            override_timestamps: Optional explicit timestamp tensor.

        Returns:
            Dict containing:
                - sequence_output: [B, T, D_model]
                - pooled_output:   [B, D_model]
                - sensor_tokens:   [B, T, S, D_model]
                - sensor_mask:     [B, T, S]
                - metadata:        dict
        """
        # 1. Input Projection
        sensor_tokens, sensor_mask, x = self.input_projection(module3_output)
        # x is [B, T, D_model]
        # sensor_tokens is [B, T, S, D_model]
        # sensor_mask is [B, T, S] boolean

        B, T, D = x.shape

        # Extract timestamps
        timestamps = override_timestamps
        if timestamps is None and "timestamps" in module3_output:
            timestamps = module3_output["timestamps"]

        # 2. Add Temporal Encoding
        x = self.temporal_encoding(x, timestamps=timestamps)

        # 3. Pass through Stack of N Hybrid Blocks
        curr_sensor_tokens = sensor_tokens
        for block in self.blocks:
            x, curr_sensor_tokens = block(
                x=x,
                sensor_tokens=curr_sensor_tokens,
                sensor_mask=sensor_mask,
            )

        # 4. Final Normalization
        sequence_output = self.final_norm(x)  # [B, T, D_model]

        # Build overall sequence valid mask [B, T] from sensor_mask [B, T, S]
        if sensor_mask is not None:
            seq_mask = sensor_mask.any(dim=-1)  # [B, T]
        else:
            seq_mask = torch.ones((B, T), dtype=torch.bool, device=x.device)

        # 5. Sequence Pooling -> [B, D_model]
        pooled_output = self.pooling(sequence_output, seq_mask=seq_mask)

        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
            "sensor_tokens": curr_sensor_tokens,
            "sensor_mask": sensor_mask,
            "sequence_mask": seq_mask,
            "metadata": {
                "backend": self.backend_name,
                "d_model": self.config.d_model,
                "num_layers": self.config.num_layers,
                "num_sensor_groups": self.config.num_sensor_groups,
                "pooling_type": self.config.pooling,
                "use_mamba": self.config.use_mamba,
                "use_sensor_attention": self.config.use_sensor_attention,
            },
        }
