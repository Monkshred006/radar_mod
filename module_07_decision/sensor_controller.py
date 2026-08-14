"""High-Level Closed-Loop Sensor Controller for Adaptive Radar Perception.

Orchestrates perception outputs -> RL state -> policy inference -> physical parameter commands.
Guarantees strict separation: RL updates never modify Mamba backbone perception weights online.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn

from module_07_decision.state_builder import DecisionStateBuilder
from module_07_decision.reward import SensorControlReward
from module_07_decision.policy_network import (
    SensorPolicyNetwork,
    GAIN_OPTIONS,
    PULSE_WIDTH_OPTIONS,
    SAMPLING_RATE_OPTIONS,
    FRAME_AVG_OPTIONS,
)


class SensorController:
    """Closed-loop sensor adaptation controller."""

    def __init__(
        self,
        policy_network: Optional[SensorPolicyNetwork] = None,
        state_builder: Optional[DecisionStateBuilder] = None,
        reward_fn: Optional[SensorControlReward] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = device
        self.state_builder = state_builder or DecisionStateBuilder()
        self.policy_network = policy_network or SensorPolicyNetwork(
            state_dim=self.state_builder.state_dim
        )
        self.policy_network.to(device)
        self.reward_fn = reward_fn or SensorControlReward()

        # Telemetry & state tracking
        self.current_sensor_params: Dict[str, Union[float, int]] = {
            "gain_db": 0.0,
            "pulse_width_us": 10.0,
            "sampling_rate_mhz": 20.0,
            "frame_averaging": 1,
            "snr_db": 15.0,
        }
        self.prev_sensor_params: Optional[Dict[str, Union[float, int]]] = None

    def step(
        self,
        perception_outputs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Union[float, int]], torch.Tensor]:
        """Execute one control step given current perception outputs.

        Args:
            perception_outputs: Dict containing 'detection', 'classification', 'anomaly', and 'pooled_latent'.
            deterministic: If True, selects argmax policy actions.

        Returns:
            Tuple of:
                - new_sensor_params: Dictionary of physical sensor parameters.
                - state_value: Estimated baseline value from critic head.
        """
        # 1. Perception outputs MUST be detached to prevent RL updating Mamba weights
        detached_outputs = {
            k: v.detach() for k, v in perception_outputs.items() if isinstance(v, torch.Tensor)
        }

        # 2. Build RL State
        state = self.state_builder.build_state(
            detached_outputs, sensor_telemetry=self.current_sensor_params, device=self.device
        )

        # 3. Policy Action Inference
        with torch.no_grad():
            actions, _, value = self.policy_network.sample_action(
                state, deterministic=deterministic
            )

        # 4. Map discrete action indices to physical parameter values
        gain_idx = int(actions["gain"][0].item())
        pw_idx = int(actions["pulse_width"][0].item())
        sr_idx = int(actions["sampling_rate"][0].item())
        avg_idx = int(actions["frame_avg"][0].item())

        self.prev_sensor_params = dict(self.current_sensor_params)
        self.current_sensor_params = {
            "gain_db": GAIN_OPTIONS[gain_idx],
            "pulse_width_us": PULSE_WIDTH_OPTIONS[pw_idx],
            "sampling_rate_mhz": SAMPLING_RATE_OPTIONS[sr_idx],
            "frame_averaging": FRAME_AVG_OPTIONS[avg_idx],
            "snr_db": self.current_sensor_params.get("snr_db", 15.0),
        }

        return self.current_sensor_params, value
