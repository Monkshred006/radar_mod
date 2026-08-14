"""Structured Application Decision Output Data Structures for Module 7.

Exposes both continuous model information (pooled_output, raw logits, calibrated probabilities,
environmental outputs) for future Module 8 (PINN + RL) state construction, as well as
application-level decisions (target_detected, anomaly_detected, combined_event_state).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import torch


@dataclass
class PhotonShieldDecisionOutput:
    """Structured application decision output produced by DecisionLogic.

    Strictly exposes both continuous model representations/logits and application decisions.

    Continuous Model Information (for downstream PINN + RL state building):
        pooled_output: Module 4 latent representation tensor [D_model] or list of floats.
        target_logits: Raw unnormalized target classification logits.
        target_probability: Calibrated target probability (max class or positive class).
        target_probabilities: List of calibrated class probabilities for target indication.
        anomaly_logits: Raw unnormalized anomaly detection logits.
        anomaly_probability: Calibrated probability of anomaly presence.
        environmental_assessment: Environmental predictions (list of float values or class name).

    Application-Level Decisions:
        target_detected: Boolean decision for target indication.
        target_class: Predicted target class index.
        target_class_name: Human-readable class name mapping.
        anomaly_detected: Boolean decision for anomaly detection.
        combined_event_state: Application state ('NORMAL', 'TARGET', 'ANOMALY', 'TARGET_AND_ANOMALY').
        raw_outputs: Dict of raw model neural output tensors.
        metadata: Extra diagnostic or streaming metadata.
    """

    # Continuous Model Information
    pooled_output: Optional[Union[torch.Tensor, List[float]]] = None
    target_logits: Optional[List[float]] = None
    target_probability: float = 0.0
    target_probabilities: List[float] = field(default_factory=list)
    anomaly_logits: Optional[List[float]] = None
    anomaly_probability: float = 0.0
    environmental_assessment: Optional[Union[List[float], int, str]] = None

    # Application-Level Decisions
    target_detected: bool = False
    target_class: int = 0
    target_class_name: str = "no_target"
    anomaly_detected: bool = False
    combined_event_state: str = "NORMAL"

    raw_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision output to a clean serializable dictionary."""
        res: Dict[str, Any] = {
            "target_detected": self.target_detected,
            "target_class": self.target_class,
            "target_class_name": self.target_class_name,
            "target_probability": round(self.target_probability, 4),
            "target_probabilities": [round(p, 4) for p in self.target_probabilities],
            "anomaly_detected": self.anomaly_detected,
            "anomaly_probability": round(self.anomaly_probability, 4),
            "environmental_assessment": self.environmental_assessment,
            "combined_event_state": self.combined_event_state,
            "metadata": self.metadata,
        }

        if self.target_logits is not None:
            res["target_logits"] = [round(l, 4) for l in self.target_logits]

        if self.anomaly_logits is not None:
            res["anomaly_logits"] = [round(l, 4) for l in self.anomaly_logits]

        if self.pooled_output is not None:
            if isinstance(self.pooled_output, torch.Tensor):
                res["pooled_output"] = [round(float(v), 4) for v in self.pooled_output.cpu().tolist()]
            else:
                res["pooled_output"] = self.pooled_output

        return res
