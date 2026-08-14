"""Sensor Control Reward Function for RL Policy.

Computes a multi-objective reward balancing:
1. Target Detection & Classification confidence (+).
2. SNR improvement (+).
3. Power / Energy consumption penalty (-).
4. Actuator jitter / parameter switching penalty (-).
"""

from __future__ import annotations

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorControlReward:
    """Multi-Objective Reward Function for radar sensor policy controller."""

    def __init__(
        self,
        weight_detection: float = 1.0,
        weight_classification: float = 0.5,
        weight_snr: float = 0.3,
        weight_power: float = 0.2,
        weight_switch: float = 0.1,
    ) -> None:
        self.w_det = weight_detection
        self.w_cls = weight_classification
        self.w_snr = weight_snr
        self.w_pwr = weight_power
        self.w_switch = weight_switch

    def compute_reward(
        self,
        perception_outputs: Dict[str, torch.Tensor],
        current_action: Dict[str, Union[float, int]],
        prev_action: Optional[Dict[str, Union[float, int]]] = None,
        snr_gain_db: float = 0.0,
    ) -> torch.Tensor:
        """Compute scalar or batched reward.

        Args:
            perception_outputs: Output dict from PhotonV0 (detection, classification, anomaly).
            current_action: Selected sensor parameters.
            prev_action: Previous sensor parameters.
            snr_gain_db: Measured SNR gain in dB.

        Returns:
            Reward tensor of shape `[B]`.
        """
        det = perception_outputs["detection"].squeeze(-1)  # [B]
        cls_logits = perception_outputs["classification"]
        cls_probs = F.softmax(cls_logits, dim=-1)
        max_cls_prob = torch.max(cls_probs, dim=-1).values  # [B]

        # 1. Perception Confidence Reward
        r_det = self.w_det * det
        r_cls = self.w_cls * max_cls_prob

        # 2. SNR Gain Reward
        r_snr = self.w_snr * (snr_gain_db / 10.0)

        # 3. Power Consumption Penalty
        # Higher sampling rate, pulse width, and frame averaging consume more MCU energy
        p_gain = max(0.0, float(current_action.get("gain_db", 0.0))) / 20.0
        p_pw = float(current_action.get("pulse_width_us", 10.0)) / 50.0
        p_sr = float(current_action.get("sampling_rate_mhz", 20.0)) / 100.0
        p_avg = float(current_action.get("frame_averaging", 1.0)) / 8.0
        power_cost = 0.25 * (p_gain + p_pw + p_sr + p_avg)
        r_pwr = -self.w_pwr * power_cost

        # 4. Switching Penalty (discourage high frequency jitter in hardware parameters)
        switch_cost = 0.0
        if prev_action is not None:
            d_gain = abs(float(current_action.get("gain_db", 0.0)) - float(prev_action.get("gain_db", 0.0))) / 20.0
            d_pw = abs(float(current_action.get("pulse_width_us", 10.0)) - float(prev_action.get("pulse_width_us", 10.0))) / 50.0
            d_sr = abs(float(current_action.get("sampling_rate_mhz", 20.0)) - float(prev_action.get("sampling_rate_mhz", 20.0))) / 100.0
            switch_cost = 0.33 * (d_gain + d_pw + d_sr)
        r_switch = -self.w_switch * switch_cost

        total_reward = r_det + r_cls + r_snr + r_pwr + r_switch
        return total_reward
