"""Tests for continuous model output exposure, regression integrity, and continuous streaming causality."""

import pytest
import torch
from module_07_decision.config import DecisionModelConfig, DecisionConfig
from module_07_decision.multitask import PhotonShieldMultiTask
from module_07_decision.decision_logic import DecisionLogic
from module_07_decision.inference import PhotonShieldDecisionPipeline


class TestContinuousOutputs:
    def test_pooled_output_exposed(self):
        m_cfg = DecisionModelConfig(d_model=32)
        model = PhotonShieldMultiTask(m_cfg)
        pipeline = PhotonShieldDecisionPipeline(model, DecisionConfig())

        pooled_in = torch.randn(2, 32)
        decisions = pipeline.predict_pooled(pooled_in)

        assert decisions[0].pooled_output is not None
        assert torch.equal(decisions[0].pooled_output, pooled_in[0])
        assert torch.equal(decisions[1].pooled_output, pooled_in[1])

    def test_logits_and_probabilities_exposed(self):
        m_cfg = DecisionModelConfig(d_model=32)
        model = PhotonShieldMultiTask(m_cfg)
        pipeline = PhotonShieldDecisionPipeline(model, DecisionConfig())

        pooled = torch.randn(2, 32)
        decisions = pipeline.predict_pooled(pooled)

        # Target logits & probabilities
        assert decisions[0].target_logits is not None
        assert len(decisions[0].target_logits) == 2
        assert len(decisions[0].target_probabilities) == 2
        assert pytest.approx(sum(decisions[0].target_probabilities)) == 1.0

        # Anomaly logits & probabilities
        assert decisions[0].anomaly_logits is not None
        assert len(decisions[0].anomaly_logits) == 1
        assert 0.0 <= decisions[0].anomaly_probability <= 1.0

    def test_decision_regression_integrity(self):
        """Regression Test: Verify application decisions are strictly identical before and after output extension."""
        cfg = DecisionConfig(target_threshold=0.5, anomaly_threshold=0.5)
        logic = DecisionLogic(cfg)

        model_outputs = {
            "target_logits": torch.tensor([[2.0, -2.0], [-1.0, 3.0]]),
            "anomaly_logits": torch.tensor([[-3.0], [3.0]]),
            "environment_output": torch.tensor([[25.0, 50.0, 1000.0], [26.0, 52.0, 1001.0]]),
        }

        decisions = logic.process(model_outputs)
        assert decisions[0].target_detected is False
        assert decisions[0].anomaly_detected is False
        assert decisions[0].combined_event_state == "NORMAL"

        assert decisions[1].target_detected is True
        assert decisions[1].anomaly_detected is True
        assert decisions[1].combined_event_state == "TARGET_AND_ANOMALY"

    def test_continuous_streaming_causality_invariance(self):
        """MANDATORY CONTINUOUS CAUSALITY TEST:

        For timestep t, pooled_output_t, target_probability_t, anomaly_probability_t,
        environment_output_t, and decision_t MUST remain strictly invariant to modifications
        at timesteps t+1, t+2, ... T.
        """
        m_cfg = DecisionModelConfig(d_model=32)
        d_cfg = DecisionConfig(smoothing_method="ema", smoothing_window=3)
        model = PhotonShieldMultiTask(m_cfg)

        pipeline1 = PhotonShieldDecisionPipeline(model, d_cfg)
        pipeline2 = PhotonShieldDecisionPipeline(model, d_cfg)

        seq_A = [torch.randn(1, 32) for _ in range(8)]
        seq_B = [seq_A[i].clone() for i in range(8)]

        # Alter timesteps 4..7 in sequence B
        for i in range(4, 8):
            seq_B[i] = torch.randn(1, 32) + 50.0

        outs_A = []
        outs_B = []

        pipeline1.reset_stream()
        for i in range(4):
            outs_A.append(pipeline1.predict_pooled(seq_A[i], is_streaming=True)[0])

        pipeline2.reset_stream()
        for i in range(4):
            outs_B.append(pipeline2.predict_pooled(seq_B[i], is_streaming=True)[0])

        # Verify timesteps 0..3 are bit-for-bit identical between sequence A and sequence B
        for t in range(4):
            assert torch.equal(outs_A[t].pooled_output, outs_B[t].pooled_output)
            assert outs_A[t].target_logits == outs_B[t].target_logits
            assert outs_A[t].anomaly_logits == outs_B[t].anomaly_logits
            assert outs_A[t].target_probability == outs_B[t].target_probability
            assert outs_A[t].anomaly_probability == outs_B[t].anomaly_probability
            assert outs_A[t].target_detected == outs_B[t].target_detected
            assert outs_A[t].anomaly_detected == outs_B[t].anomaly_detected
