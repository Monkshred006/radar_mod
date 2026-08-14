"""Unit tests for RL decision layer and sensor controller."""

import pytest
import torch

from module_07_decision.state_builder import DecisionStateBuilder
from module_07_decision.reward import SensorControlReward
from module_07_decision.policy_network import SensorPolicyNetwork
from module_07_decision.sensor_controller import SensorController


class TestRLDecisionLayer:
    """Test suite for RL decision modules."""

    def test_decision_state_builder(self):
        builder = DecisionStateBuilder(
            include_latent=True,
            latent_dim=64,
            latent_summary_dim=8,
            num_classes=4,
            include_telemetry=True,
        )
        B = 2
        perception_outputs = {
            "detection": torch.tensor([[0.9], [0.1]]),
            "classification": torch.randn(B, 4),
            "anomaly": torch.tensor([[0.05], [0.8]]),
            "pooled_latent": torch.randn(B, 64),
        }
        state = builder.build_state(perception_outputs)
        assert state.shape == (B, builder.state_dim)
        assert not torch.isnan(state).any()

    def test_sensor_control_reward(self):
        reward_fn = SensorControlReward()
        perception_outputs = {
            "detection": torch.tensor([[0.9]]),
            "classification": torch.tensor([[0.1, 0.2, 0.9, 0.1]]),
            "anomaly": torch.tensor([[0.0]]),
        }
        action = {"gain_db": 6.0, "pulse_width_us": 10.0, "sampling_rate_mhz": 20.0, "frame_averaging": 1}
        reward = reward_fn.compute_reward(perception_outputs, current_action=action, snr_gain_db=5.0)
        assert reward.shape == (1,)
        assert isinstance(reward.item(), float)

    def test_policy_network_sample_action(self):
        state_dim = 19
        policy = SensorPolicyNetwork(state_dim=state_dim, hidden_dim=32)
        state = torch.randn(2, state_dim)

        actions, log_probs, value = policy.sample_action(state, deterministic=True)
        assert "gain" in actions
        assert "pulse_width" in actions
        assert "sampling_rate" in actions
        assert "frame_avg" in actions
        assert value.shape == (2, 1)

    def test_sensor_controller_step_isolation(self):
        controller = SensorController()
        perception_outputs = {
            "detection": torch.tensor([[0.85]], requires_grad=True),
            "classification": torch.randn(1, 4, requires_grad=True),
            "anomaly": torch.tensor([[0.02]], requires_grad=True),
            "pooled_latent": torch.randn(1, 64, requires_grad=True),
        }

        # Step controller
        new_params, value = controller.step(perception_outputs, deterministic=True)
        assert "gain_db" in new_params
        assert "pulse_width_us" in new_params
        assert "sampling_rate_mhz" in new_params
        assert "frame_averaging" in new_params

        # Ensure no gradients flow into perception outputs
        assert perception_outputs["detection"].grad is None
        assert perception_outputs["pooled_latent"].grad is None
