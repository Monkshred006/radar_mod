"""Tests for streaming inference decision pipeline and MANDATORY STREAMING CAUSALITY TEST."""

import pytest
import torch
from module_07_decision.config import DecisionModelConfig, DecisionConfig
from module_07_decision.multitask import PhotonShieldMultiTask
from module_07_decision.inference import PhotonShieldDecisionPipeline


class TestInferencePipeline:
    def test_pooled_output_predict(self):
        m_cfg = DecisionModelConfig(d_model=32)
        d_cfg = DecisionConfig()
        model = PhotonShieldMultiTask(m_cfg)
        pipeline = PhotonShieldDecisionPipeline(model, d_cfg)

        pooled = torch.randn(4, 32)
        decisions = pipeline.predict_pooled(pooled)
        assert len(decisions) == 4

    def test_mandatory_streaming_causality(self):
        """MANDATORY STREAMING CAUSALITY TEST:

        For a decision at time t: decision(t) must depend ONLY on predictions[0:t].
        Modifying predictions at t+1, t+2, ... MUST NOT change decision(t).
        """
        m_cfg = DecisionModelConfig(d_model=32)
        d_cfg = DecisionConfig(
            smoothing_method="ema",
            smoothing_window=5,
            minimum_consecutive_detections=2,
            hysteresis_enabled=True,
        )
        model = PhotonShieldMultiTask(m_cfg)
        pipeline1 = PhotonShieldDecisionPipeline(model, d_cfg)
        pipeline2 = PhotonShieldDecisionPipeline(model, d_cfg)

        # Sequence of 10 time steps
        t_seq_A = [torch.randn(1, 32) for _ in range(10)]
        t_seq_B = [t_seq_A[i].clone() for i in range(10)]

        # Alter time steps 5..9 in sequence B
        for i in range(5, 10):
            t_seq_B[i] = torch.randn(1, 32) + 100.0

        decisions_A = []
        decisions_B = []

        # Run pipeline 1 step-by-step up to time t=4
        pipeline1.reset_stream()
        for i in range(5):
            d = pipeline1.predict_pooled(t_seq_A[i], is_streaming=True)[0]
            decisions_A.append(d)

        # Run pipeline 2 step-by-step up to time t=4 using sequence B
        pipeline2.reset_stream()
        for i in range(5):
            d = pipeline2.predict_pooled(t_seq_B[i], is_streaming=True)[0]
            decisions_B.append(d)

        # decision(t) at t=4 must be STRICTLY IDENTICAL between A and B
        for t in range(5):
            assert decisions_A[t].target_detected == decisions_B[t].target_detected
            assert decisions_A[t].anomaly_detected == decisions_B[t].anomaly_detected
            assert pytest.approx(decisions_A[t].target_probability, abs=1e-5) == decisions_B[t].target_probability
            assert pytest.approx(decisions_A[t].anomaly_probability, abs=1e-5) == decisions_B[t].anomaly_probability
