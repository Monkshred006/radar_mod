"""Tests for DecisionLogic, thresholding, hysteresis, and probability conversion."""

import pytest
import torch
from module_07_decision.config import DecisionConfig
from module_07_decision.decision_logic import DecisionLogic


class TestDecisionLogic:
    def test_basic_processing(self):
        cfg = DecisionConfig(target_threshold=0.5, anomaly_threshold=0.5)
        logic = DecisionLogic(cfg)

        model_outputs = {
            "target_logits": torch.tensor([[1.0, -1.0], [-1.0, 2.0]]),  # target 0 vs target 1
            "anomaly_logits": torch.tensor([[-2.0], [2.0]]),             # low vs high anomaly
            "environment_output": torch.tensor([[20.5, 55.0, 1013.2], [22.0, 50.0, 1012.0]]),
        }

        decisions = logic.process(model_outputs)
        assert len(decisions) == 2

        # Sample 0
        assert decisions[0].target_detected is False
        assert decisions[0].anomaly_detected is False
        assert decisions[0].combined_event_state == "NORMAL"

        # Sample 1
        assert decisions[1].target_detected is True
        assert decisions[1].anomaly_detected is True
        assert decisions[1].combined_event_state == "TARGET_AND_ANOMALY"

    def test_hysteresis_logic(self):
        cfg = DecisionConfig(
            hysteresis_enabled=True,
            hysteresis_on_threshold=0.7,
            hysteresis_off_threshold=0.3,
        )
        logic = DecisionLogic(cfg)

        # Probabilities: 0.6 (below 0.7 -> False), 0.8 (above 0.7 -> True), 0.5 (above 0.3 -> remains True), 0.2 (below 0.3 -> False)
        logits_seq = [-0.4, 1.5, 0.0, -1.5]  # sigmoids approx: 0.40, 0.82, 0.50, 0.18

        results = []
        for l in logits_seq:
            m_out = {"anomaly_logits": torch.tensor([[l]])}
            d = logic.process(m_out)[0]
            results.append(d.anomaly_detected)

        assert results == [False, True, True, False]
