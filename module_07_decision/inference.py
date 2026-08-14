"""Real-Time Streaming Decision Pipeline for Module 7."""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn

from module_07_decision.config import DecisionModelConfig, DecisionConfig
from module_07_decision.multitask import PhotonShieldMultiTask
from module_07_decision.decision_logic import DecisionLogic
from module_07_decision.outputs import PhotonShieldDecisionOutput


class PhotonShieldDecisionPipeline:
    """Integrated inference decision pipeline for PhotonShield.

    Connects Module 4 / Module 6 engine representation (pooled_output) to
    PhotonShieldMultiTask heads and DecisionLogic layer. Exposes continuous
    latent representations, logits, probabilities, and application decisions.

    Args:
        multi_task_model: PhotonShieldMultiTask instance.
        decision_config: DecisionConfig.
        engine: Optional Module 4 or Module 6 backbone engine.
    """

    def __init__(
        self,
        multi_task_model: PhotonShieldMultiTask,
        decision_config: DecisionConfig,
        engine: Optional[nn.Module] = None,
    ):
        self.multi_task_model = multi_task_model
        self.decision_config = decision_config
        self.engine = engine
        self.decision_logic = DecisionLogic(decision_config)

    def reset_stream(self) -> None:
        """Reset internal causal streaming state."""
        self.decision_logic.reset_streaming_state()

    def predict_pooled(
        self,
        pooled_output: torch.Tensor,
        is_streaming: bool = False,
    ) -> List[PhotonShieldDecisionOutput]:
        """Run multi-task heads and decision logic directly on pooled_output tensor [B, D_model].

        Args:
            pooled_output: Representation tensor [B, d_model].
            is_streaming: Whether to update causal decision smoothing buffers frame-by-frame.

        Returns:
            List of structured PhotonShieldDecisionOutput objects.
        """
        self.multi_task_model.eval()
        with torch.no_grad():
            model_outputs = self.multi_task_model(pooled_output)
            return self.decision_logic.process(
                model_outputs=model_outputs,
                pooled_output=pooled_output,
                is_streaming=is_streaming,
            )

    def predict_sample(
        self,
        sample_batch: Dict[str, Any],
        is_streaming: bool = False,
    ) -> List[PhotonShieldDecisionOutput]:
        """Run end-to-end inference from sample dict using Module 4/6 engine backbone.

        Args:
            sample_batch: Module 3 fused sample dictionary.
            is_streaming: Whether to operate in causal streaming state mode.

        Returns:
            List of structured PhotonShieldDecisionOutput objects.
        """
        if self.engine is None:
            raise RuntimeError("Engine backbone must be provided to predict_sample")

        self.engine.eval()
        self.multi_task_model.eval()

        with torch.no_grad():
            m_out = self.engine(sample_batch)
            pooled_out = m_out["pooled_output"]
            model_outputs = self.multi_task_model(pooled_out)
            return self.decision_logic.process(
                model_outputs=model_outputs,
                pooled_output=pooled_out,
                is_streaming=is_streaming,
            )
