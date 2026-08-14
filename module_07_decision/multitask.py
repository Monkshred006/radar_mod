"""PhotonShield Multi-Task Head Container and Multi-Task Loss for Module 7."""

from __future__ import annotations
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_07_decision.config import DecisionModelConfig
from module_07_decision.target_head import TargetHead
from module_07_decision.anomaly_head import AnomalyHead
from module_07_decision.environmental_head import EnvironmentalHead


class PhotonShieldMultiTask(nn.Module):
    """PhotonShield Multi-Task Prediction Module.

    Attaches Target Indication, Anomaly Detection, and Environmental Assessment heads
    to Module 4's pooled_output representation.

    Args:
        config: DecisionModelConfig.
    """

    def __init__(self, config: DecisionModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.target_head: Optional[TargetHead] = None
        self.anomaly_head: Optional[AnomalyHead] = None
        self.environmental_head: Optional[EnvironmentalHead] = None

        if config.enable_target:
            self.target_head = TargetHead.from_config(config)

        if config.enable_anomaly:
            self.anomaly_head = AnomalyHead.from_config(config)

        if config.enable_environment:
            self.environmental_head = EnvironmentalHead.from_config(config)

    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass across all enabled task heads.

        Args:
            pooled_output: Latent representation tensor [B, d_model] from Module 4.

        Returns:
            Dictionary containing model outputs for enabled tasks:
                - 'target_logits': [B, num_target_classes] (if enabled)
                - 'anomaly_logits': [B, 1] (if enabled)
                - 'environment_output': [B, N_env] (if enabled)
        """
        outputs: Dict[str, torch.Tensor] = {}

        if self.target_head is not None:
            outputs["target_logits"] = self.target_head(pooled_output)

        if self.anomaly_head is not None:
            outputs["anomaly_logits"] = self.anomaly_head(pooled_output)

        if self.environmental_head is not None:
            outputs["environment_output"] = self.environmental_head(pooled_output)

        return outputs


class MultiTaskDecisionLoss(nn.Module):
    """Multi-Task Loss for PhotonShield Decision Heads with missing label support.

    Total Loss:
        L_total = λ_target * L_target + λ_anomaly * L_anomaly + λ_env * L_env

    Supports missing task labels without converting missing labels to false zero targets.

    Args:
        config: DecisionModelConfig.
        weight_target: Weight λ for target indication loss.
        weight_anomaly: Weight λ for anomaly detection loss.
        weight_environment: Weight λ for environmental assessment loss.
        pos_weight_anomaly: Optional positive class weight for BCE anomaly loss.
        environment_loss_fn: Loss function for environmental regression ('mse', 'l1', 'huber') or 'ce' for classification.
    """

    def __init__(
        self,
        config: DecisionModelConfig,
        weight_target: float = 1.0,
        weight_anomaly: float = 1.0,
        weight_environment: float = 1.0,
        pos_weight_anomaly: Optional[float] = None,
        environment_loss_fn: str = "mse",
    ):
        super().__init__()
        self.config = config
        self.weight_target = weight_target
        self.weight_anomaly = weight_anomaly
        self.weight_environment = weight_environment
        self.pos_weight_anomaly = pos_weight_anomaly
        self.environment_loss_fn = environment_loss_fn

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss with missing label masking.

        Args:
            model_outputs: Dictionary from PhotonShieldMultiTask forward pass.
            targets: Dictionary of target labels:
                - 'target_labels': [B] long tensor or None
                - 'anomaly_labels': [B] float tensor or [B, 1] or None
                - 'environment_labels': [B, N_env] float tensor or None

        Returns:
            Dictionary containing 'loss' (total scalar) and individual task loss terms.
        """
        device = next(p for p in model_outputs.values() if isinstance(p, torch.Tensor)).device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict: Dict[str, torch.Tensor] = {}

        # 1. Target Indication Loss (CrossEntropyLoss)
        if "target_logits" in model_outputs and "target_labels" in targets:
            t_labels = targets["target_labels"]
            if t_labels is not None:
                # Mask out invalid (-100 or negative) missing label markers if present
                mask = t_labels >= 0
                if mask.any():
                    t_loss = F.cross_entropy(model_outputs["target_logits"][mask], t_labels[mask].long())
                    loss_dict["loss_target"] = t_loss
                    total_loss += self.weight_target * t_loss

        # 2. Anomaly Detection Loss (BCEWithLogitsLoss)
        if "anomaly_logits" in model_outputs and "anomaly_labels" in targets:
            a_labels = targets["anomaly_labels"]
            if a_labels is not None:
                a_logits = model_outputs["anomaly_logits"].view(-1)
                a_targets = a_labels.view(-1).float()
                mask = a_targets >= 0.0  # -1 is missing label
                if mask.any():
                    pos_w = (
                        torch.tensor([self.pos_weight_anomaly], device=device)
                        if self.pos_weight_anomaly is not None
                        else None
                    )
                    a_loss = F.binary_cross_entropy_with_logits(
                        a_logits[mask], a_targets[mask], pos_weight=pos_w
                    )
                    loss_dict["loss_anomaly"] = a_loss
                    total_loss += self.weight_anomaly * a_loss

        # 3. Environmental Assessment Loss (MSE/L1/Huber or CE)
        if "environment_output" in model_outputs and "environment_labels" in targets:
            e_labels = targets["environment_labels"]
            if e_labels is not None:
                e_out = model_outputs["environment_output"]
                if self.config.environment_mode == "regression":
                    # Mask out NaNs or negative invalid markers if mask is provided
                    mask = ~torch.isnan(e_labels)
                    if "environment_mask" in targets and targets["environment_mask"] is not None:
                        mask = mask & targets["environment_mask"].bool()
                    if mask.any():
                        if self.environment_loss_fn == "l1":
                            e_loss = F.l1_loss(e_out[mask], e_labels[mask])
                        elif self.environment_loss_fn == "huber":
                            e_loss = F.huber_loss(e_out[mask], e_labels[mask])
                        else:
                            e_loss = F.mse_loss(e_out[mask], e_labels[mask])
                        loss_dict["loss_environment"] = e_loss
                        total_loss += self.weight_environment * e_loss
                else:
                    # Classification mode
                    mask = e_labels >= 0
                    if mask.any():
                        e_loss = F.cross_entropy(e_out[mask], e_labels[mask].long())
                        loss_dict["loss_environment"] = e_loss
                        total_loss += self.weight_environment * e_loss

        loss_dict["loss"] = total_loss
        return loss_dict
