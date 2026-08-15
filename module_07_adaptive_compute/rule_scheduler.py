"""Deterministic Rule-Based Scheduler for PhotonShield V3 Adaptive Compute.

Derives diffusion reverse step budget N in {5, 10, 20, 50} directly from state vector
before launching the diffusion trajectory.
"""

from __future__ import annotations

from typing import Union, List
import numpy as np
import torch

from module_07_adaptive_compute.action_space import ACTIONS


class RuleBasedDiffusionScheduler:
    """Rule-based deterministic controller for adaptive diffusion steps."""

    def __init__(
        self,
        gap_thresh: float = 0.20,        # >3 missing frames in length
        kin_thresh: float = 0.12,        # normalized kinematic mismatch
        snr_min_thresh: float = 0.08,    # very low SNR floor
        extreme_gap_thresh: float = 0.45, # >7 consecutive missing frames
    ) -> None:
        self.gap_thresh = gap_thresh
        self.kin_thresh = kin_thresh
        self.snr_min_thresh = snr_min_thresh
        self.extreme_gap_thresh = extreme_gap_thresh

    def predict_action(
        self,
        state: Union[np.ndarray, torch.Tensor, dict],
    ) -> int:
        """Select discrete diffusion steps N in {5, 10, 20, 50} from state.

        Args:
            state: Normalized 9D state vector [snr, obs_ratio, gap_length, est_range, est_vel,
                   kin_res, energy_res, r_unc, v_unc] or dictionary.

        Returns:
            Selected diffusion step count (5, 10, 20, or 50).
        """
        if isinstance(state, dict):
            gap_len = float(state.get("gap_length", 0.0))
            kin_res = float(state.get("kin_residual", 0.0))
            snr = float(state.get("snr_quality", 0.5))
            r_unc = float(state.get("r_uncertainty", 0.0))
        elif isinstance(state, torch.Tensor):
            s = state.cpu().numpy().flatten()
            snr = float(s[0])
            gap_len = float(s[2])
            kin_res = float(s[5])
            r_unc = float(s[7])
        else:
            s = np.asarray(state).flatten()
            snr = float(s[0])
            gap_len = float(s[2])
            kin_res = float(s[5])
            r_unc = float(s[7])

        # Decision Rule Cascade:
        # 1. Extreme gap & high uncertainty -> 20 steps
        if gap_len >= self.extreme_gap_thresh and (r_unc > 0.35 or kin_res > 0.30):
            return 20

        # 2. Moderate gap length or elevated kinematic residual or low SNR -> 10 steps
        if gap_len >= self.gap_thresh or kin_res >= self.kin_thresh or snr < self.snr_min_thresh:
            return 10

        # 3. Default nominal / low-corruption regime -> 5 steps
        return 5

    def predict_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
    ) -> List[int]:
        """Predict actions for a batch of state vectors."""
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        return [self.predict_action(s) for s in states]
