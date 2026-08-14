"""Module 7: PhotonShield Decision / Task Output Layer."""

from module_07_decision.config import DecisionModelConfig, DecisionConfig
from module_07_decision.heads import BaseTaskHead
from module_07_decision.target_head import TargetHead
from module_07_decision.anomaly_head import AnomalyHead
from module_07_decision.environmental_head import EnvironmentalHead
from module_07_decision.multitask import PhotonShieldMultiTask, MultiTaskDecisionLoss
from module_07_decision.outputs import PhotonShieldDecisionOutput
from module_07_decision.decision_logic import DecisionLogic
from module_07_decision.thresholds import analyze_validation_thresholds
from module_07_decision.calibration import TemperatureScaler
from module_07_decision.checkpointing import save_decision_checkpoint, load_decision_checkpoint
from module_07_decision.inference import PhotonShieldDecisionPipeline

__all__ = [
    "DecisionModelConfig",
    "DecisionConfig",
    "BaseTaskHead",
    "TargetHead",
    "AnomalyHead",
    "EnvironmentalHead",
    "PhotonShieldMultiTask",
    "MultiTaskDecisionLoss",
    "PhotonShieldDecisionOutput",
    "DecisionLogic",
    "analyze_validation_thresholds",
    "TemperatureScaler",
    "save_decision_checkpoint",
    "load_decision_checkpoint",
    "PhotonShieldDecisionPipeline",
]
