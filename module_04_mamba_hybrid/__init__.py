"""Module 4 — PhotonShield Mamba-Hybrid Engine.

Exposes:
- `MambaHybridConfig`, `TaskHeadConfig`
- `PhotonMambaHybrid`
- `ClassificationHead`, `RegressionHead`, `MultiTaskHead`
- `get_loss_fn`
- `save_checkpoint`, `load_checkpoint`
- `profile_model`, `count_parameters`
"""

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.input_projection import SensorTokenProjection
from module_04_mamba_hybrid.temporal_encoding import TemporalEncoding
from module_04_mamba_hybrid.mamba_block import MambaTemporalBranch
from module_04_mamba_hybrid.sensor_interaction import CrossSensorInteractionBranch
from module_04_mamba_hybrid.hybrid_block import HybridBlock
from module_04_mamba_hybrid.pooling import SequencePooling
from module_04_mamba_hybrid.engine import PhotonMambaHybrid, EngineOutput
from module_04_mamba_hybrid.heads import ClassificationHead, RegressionHead, MultiTaskHead
from module_04_mamba_hybrid.losses import get_loss_fn, MultiTaskLoss
from module_04_mamba_hybrid.checkpointing import save_checkpoint, load_checkpoint
from module_04_mamba_hybrid.profiling import profile_model, count_parameters

__all__ = [
    "MambaHybridConfig",
    "TaskHeadConfig",
    "SensorTokenProjection",
    "TemporalEncoding",
    "MambaTemporalBranch",
    "CrossSensorInteractionBranch",
    "HybridBlock",
    "SequencePooling",
    "PhotonMambaHybrid",
    "EngineOutput",
    "ClassificationHead",
    "RegressionHead",
    "MultiTaskHead",
    "get_loss_fn",
    "MultiTaskLoss",
    "save_checkpoint",
    "load_checkpoint",
    "profile_model",
    "count_parameters",
]
