"""Deterministic Decision Layer and Causal Decision Smoothing for Module 7."""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Deque, Union
from collections import deque
import torch
import torch.nn.functional as F

from module_07_decision.config import DecisionConfig
from module_07_decision.outputs import PhotonShieldDecisionOutput


class DecisionLogic:
    """Deterministic Decision Layer converting raw neural model outputs to application decisions.

    Strictly exposes both continuous model outputs (pooled_output, logits, probabilities) and application decisions.
    Supports temperature calibration, configurable thresholds, dual-threshold hysteresis,
    and causal decision smoothing.

    Args:
        config: DecisionConfig.
    """

    def __init__(self, config: DecisionConfig):
        config.validate()
        self.config = config

        # Streaming state buffers for causal smoothing (strictly past-to-present)
        self.target_prob_buffer: Deque[float] = deque(maxlen=config.smoothing_window)
        self.anomaly_prob_buffer: Deque[float] = deque(maxlen=config.smoothing_window)
        self.target_decision_buffer: Deque[bool] = deque(maxlen=config.smoothing_window)
        self.anomaly_decision_buffer: Deque[bool] = deque(maxlen=config.smoothing_window)

        # Hysteresis state tracking
        self.target_hysteresis_state: bool = False
        self.anomaly_hysteresis_state: bool = False

    def reset_streaming_state(self) -> None:
        """Reset all streaming buffers and hysteresis states."""
        self.target_prob_buffer.clear()
        self.anomaly_prob_buffer.clear()
        self.target_decision_buffer.clear()
        self.anomaly_decision_buffer.clear()
        self.target_hysteresis_state = False
        self.anomaly_hysteresis_state = False

    def process(
        self,
        model_outputs: Dict[str, torch.Tensor],
        pooled_output: Optional[torch.Tensor] = None,
        is_streaming: bool = False,
    ) -> List[PhotonShieldDecisionOutput]:
        """Process model outputs into continuous information and structured application decisions.

        Args:
            model_outputs: Dict containing 'target_logits', 'anomaly_logits', 'environment_output'.
            pooled_output: Optional Module 4 latent representation tensor [B, D_model].
            is_streaming: If True, updates internal causal state buffers frame-by-frame.

        Returns:
            List of PhotonShieldDecisionOutput objects for each sample in the batch [B].
        """
        device = next(p for p in model_outputs.values() if isinstance(p, torch.Tensor)).device
        batch_size = next(p for p in model_outputs.values() if isinstance(p, torch.Tensor)).shape[0]

        decision_outputs: List[PhotonShieldDecisionOutput] = []

        for b in range(batch_size):
            out = PhotonShieldDecisionOutput()
            out.raw_outputs = {k: v[b].detach().cpu() for k, v in model_outputs.items()}

            # 0. Preserve Module 4 Latent Representation
            if pooled_output is not None:
                out.pooled_output = pooled_output[b].detach()
            elif "pooled_output" in model_outputs:
                out.pooled_output = model_outputs["pooled_output"][b].detach()

            # 1. Target Indication Decision & Continuous Logits/Probabilities
            if "target_logits" in model_outputs:
                raw_t_logits = model_outputs["target_logits"][b].detach().cpu()
                out.target_logits = raw_t_logits.tolist()

                t_logits = model_outputs["target_logits"][b]

                # Temperature scaling calibration
                if self.config.calibration_method == "temperature":
                    t_logits = t_logits / max(self.config.temperature_target, 1e-4)

                t_probs = F.softmax(t_logits, dim=-1).cpu().tolist()
                out.target_probabilities = t_probs

                if len(t_probs) == 2:
                    # Binary target classification
                    raw_prob = t_probs[1]
                    pred_class = 1 if raw_prob >= self.config.target_threshold else 0
                else:
                    # Multi-class classification
                    pred_class = int(torch.argmax(t_logits).item())
                    raw_prob = t_probs[pred_class]

                smoothed_prob = self._apply_prob_smoothing(raw_prob, self.target_prob_buffer) if is_streaming else raw_prob
                raw_decision = (smoothed_prob >= self.config.target_threshold) if len(t_probs) == 2 else (pred_class > 0 and smoothed_prob >= self.config.target_threshold)

                if self.config.hysteresis_enabled:
                    raw_decision = self._apply_hysteresis(
                        smoothed_prob,
                        self.target_hysteresis_state,
                        self.config.hysteresis_on_threshold,
                        self.config.hysteresis_off_threshold,
                    )
                    self.target_hysteresis_state = raw_decision

                final_target_decision = self._apply_consecutive_smoothing(raw_decision, self.target_decision_buffer) if is_streaming else raw_decision

                out.target_detected = final_target_decision
                out.target_class = pred_class if final_target_decision else 0
                out.target_class_name = self.config.target_class_labels.get(out.target_class, f"class_{out.target_class}")
                out.target_probability = smoothed_prob

            # 2. Anomaly Detection Decision & Continuous Logits/Probabilities
            if "anomaly_logits" in model_outputs:
                raw_a_logits = model_outputs["anomaly_logits"][b].detach().cpu().squeeze()
                out.anomaly_logits = [float(raw_a_logits.item())] if raw_a_logits.ndim == 0 else raw_a_logits.tolist()

                a_logits = model_outputs["anomaly_logits"][b].squeeze()

                if self.config.calibration_method == "temperature":
                    a_logits = a_logits / max(self.config.temperature_anomaly, 1e-4)

                a_prob = float(torch.sigmoid(a_logits).item())

                smoothed_a_prob = self._apply_prob_smoothing(a_prob, self.anomaly_prob_buffer) if is_streaming else a_prob
                raw_a_decision = smoothed_a_prob >= self.config.anomaly_threshold

                if self.config.hysteresis_enabled:
                    raw_a_decision = self._apply_hysteresis(
                        smoothed_a_prob,
                        self.anomaly_hysteresis_state,
                        self.config.hysteresis_on_threshold,
                        self.config.hysteresis_off_threshold,
                    )
                    self.anomaly_hysteresis_state = raw_a_decision

                final_anomaly_decision = self._apply_consecutive_smoothing(raw_a_decision, self.anomaly_decision_buffer) if is_streaming else raw_a_decision

                out.anomaly_detected = final_anomaly_decision
                out.anomaly_probability = smoothed_a_prob

            # 3. Environmental Assessment Output
            if "environment_output" in model_outputs:
                e_val = model_outputs["environment_output"][b].detach().cpu()
                if e_val.ndim == 0 or e_val.numel() == 1:
                    out.environmental_assessment = float(e_val.item())
                else:
                    out.environmental_assessment = [round(float(v), 4) for v in e_val.tolist()]

            # 4. Event Combination Logic
            if self.config.event_combination_enabled:
                if out.target_detected and out.anomaly_detected:
                    out.combined_event_state = "TARGET_AND_ANOMALY"
                elif out.target_detected:
                    out.combined_event_state = "TARGET"
                elif out.anomaly_detected:
                    out.combined_event_state = "ANOMALY"
                else:
                    out.combined_event_state = "NORMAL"

            decision_outputs.append(out)

        return decision_outputs

    def _apply_prob_smoothing(self, prob: float, buffer: Deque[float]) -> float:
        """Apply causal probability smoothing (EMA or windowed mean)."""
        buffer.append(prob)
        if self.config.smoothing_method == "ema":
            alpha = 2.0 / (len(buffer) + 1)
            ema = buffer[0]
            for p in list(buffer)[1:]:
                ema = alpha * p + (1.0 - alpha) * ema
            return ema
        elif self.config.smoothing_method == "majority_vote":
            return sum(buffer) / len(buffer)
        return prob

    def _apply_consecutive_smoothing(self, decision: bool, buffer: Deque[bool]) -> bool:
        """Apply minimum consecutive detections constraint strictly causally."""
        buffer.append(decision)
        if self.config.minimum_consecutive_detections <= 1:
            return decision

        if len(buffer) < self.config.minimum_consecutive_detections:
            return False

        recent = list(buffer)[-self.config.minimum_consecutive_detections:]
        return all(recent)

    def _apply_hysteresis(self, prob: float, current_state: bool, on_thresh: float, off_thresh: float) -> bool:
        """Apply dual-threshold hysteresis."""
        if current_state:
            return prob >= off_thresh
        else:
            return prob >= on_thresh
