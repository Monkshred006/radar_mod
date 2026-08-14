"""Sensor-Aware Tokenization layer for Mamba-Hybrid engine.

Converts fused multi-sensor feature matrices into structured 4D tokens:
    [B, T, S, D_features]

where:
- B = Batch size
- T = Timesteps
- S = Sensor groups (0: optical, 1: environment, 2: motion, 3: distance, 4: quality)
- D_features = Feature dimension per sensor group

Features are padded to max D_features if pad_to_max_dim=True, accompanied
by a boolean group mask [B, T, S, D_features] or [B, T, S].

Feature dimension D_features is intentionally NOT hard-coded to Mamba model
dimension D_model (Module 4 performs the learnable projection D_features -> D_model).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from module_03_sensor_fusion.config import TokenizerConfig


class SensorAwareTokenizer:
    """Tokenizer creating sensor-aware 4D tokens."""

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()

    def build_tokens_single(
        self,
        fused_matrix: np.ndarray,
        group_map: Dict[str, Tuple[int, int]],
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Construct tokens for a single sequence [T, S, D_features].

        Args:
            fused_matrix: Fused features [T, F_total]
            group_map: Dict group_name -> (start_col, end_col)
            dtype: Target torch dtype

        Returns:
            Tuple of:
            - tokens_tensor: torch.Tensor [T, S, D_max]
            - group_mask: torch.Tensor [T, S, D_max] (bool: True = valid feature, False = padding)
            - active_groups: List of group names in order S
        """
        T, F = fused_matrix.shape
        order = self.config.explicit_group_order

        # Filter active groups present in group_map
        active_groups = [g for g in order if g in group_map]
        if not active_groups:
            # Fallback if no explicit groups match
            active_groups = list(group_map.keys())

        S = len(active_groups)
        if S == 0:
            tokens = torch.zeros((T, 0, 0), dtype=dtype)
            mask = torch.zeros((T, 0, 0), dtype=torch.bool)
            return tokens, mask, []

        # Find max feature dimension across groups
        group_vectors = []
        max_d = 0
        for grp in active_groups:
            start, end = group_map[grp]
            vec = fused_matrix[:, start:end]  # [T, D_grp]
            group_vectors.append(vec)
            max_d = max(max_d, end - start)

        # Build padded 3D array [T, S, max_d]
        tokens_np = np.zeros((T, S, max_d), dtype=np.float64)
        mask_np = np.zeros((T, S, max_d), dtype=bool)

        for s_idx, vec in enumerate(group_vectors):
            d_grp = vec.shape[1]
            tokens_np[:, s_idx, :d_grp] = vec
            mask_np[:, s_idx, :d_grp] = True

        tokens_tensor = torch.from_numpy(tokens_np).to(dtype=dtype)
        mask_tensor = torch.from_numpy(mask_np).to(dtype=torch.bool)

        return tokens_tensor, mask_tensor, active_groups

    def build_tokens_batched(
        self,
        fused_matrices: List[np.ndarray],
        group_maps: List[Dict[str, Tuple[int, int]]],
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Construct tokens for a batch of sequences [B, T, S, D_max].

        Args:
            fused_matrices: List of [T, F] matrices length B
            group_maps: List of group_map dicts length B
            dtype: Target torch dtype

        Returns:
            Tuple of:
            - batched_tokens: torch.Tensor [B, T, S, D_max]
            - batched_mask: torch.Tensor [B, T, S, D_max]
            - active_groups: List of group names
        """
        B = len(fused_matrices)
        single_tokens = []
        single_masks = []
        groups = []

        for i in range(B):
            tok, mask, active = self.build_tokens_single(
                fused_matrices[i], group_maps[i], dtype=dtype
            )
            single_tokens.append(tok)
            single_masks.append(mask)
            groups = active

        # Stack into batch dimension B
        batched_tokens = torch.stack(single_tokens, dim=0)  # [B, T, S, D_max]
        batched_mask = torch.stack(single_masks, dim=0)      # [B, T, S, D_max]

        return batched_tokens, batched_mask, groups
