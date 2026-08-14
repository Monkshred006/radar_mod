"""Tests for structured application decision output serialization."""

import pytest
from module_07_decision.outputs import PhotonShieldDecisionOutput


class TestOutputs:
    def test_to_dict_conversion(self):
        out = PhotonShieldDecisionOutput(
            target_detected=True,
            target_class=1,
            target_class_name="target",
            target_probability=0.8851,
            target_probabilities=[0.1149, 0.8851],
            anomaly_detected=False,
            anomaly_probability=0.05,
            environmental_assessment=[21.5, 60.0, 1013.25],
            combined_event_state="TARGET",
        )

        d = out.to_dict()
        assert d["target_detected"] is True
        assert d["target_class"] == 1
        assert d["combined_event_state"] == "TARGET"
        assert len(d["target_probabilities"]) == 2
