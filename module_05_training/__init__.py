"""Module 5: FP32 Training + Evaluation Pipeline for PhotonShield AI."""

from module_05_training.config import TrainingConfig
from module_05_training.reproducibility import set_seed, get_seed_state, restore_seed_state
from module_05_training.target_adapter import (
    TargetAdapter,
    SyntheticRegressionAdapter,
    SyntheticClassificationAdapter,
    get_target_adapter,
)
from module_05_training.dataset import (
    SceneFeatureCache,
    PhotonShieldDataset,
    collate_module3,
    make_synthetic_scene_cache,
)
from module_05_training.losses import get_loss_fn, TrainingNaNError, WeightedMultiTaskLoss
from module_05_training.metrics import MetricsTracker, MultiTaskMetricsTracker
from module_05_training.optimizer import get_optimizer
from module_05_training.scheduler import get_scheduler
from module_05_training.early_stopping import EarlyStopping
from module_05_training.checkpointing import save_checkpoint, load_checkpoint
from module_05_training.logging_utils import ExperimentLogger
from module_05_training.trainer import Trainer
from module_05_training.evaluator import Evaluator
from module_05_training.profiling import profile_model
from module_05_training.experiment import ExperimentRunner, ABLATION_VARIANTS
from module_05_training.noise_scheduler import NoiseScheduler
from module_05_training.diffusion_auxiliary import DiffusionAuxiliary

__all__ = [
    "TrainingConfig",
    "set_seed",
    "get_seed_state",
    "restore_seed_state",
    "TargetAdapter",
    "SyntheticRegressionAdapter",
    "SyntheticClassificationAdapter",
    "get_target_adapter",
    "SceneFeatureCache",
    "PhotonShieldDataset",
    "collate_module3",
    "make_synthetic_scene_cache",
    "get_loss_fn",
    "TrainingNaNError",
    "WeightedMultiTaskLoss",
    "MetricsTracker",
    "MultiTaskMetricsTracker",
    "get_optimizer",
    "get_scheduler",
    "EarlyStopping",
    "save_checkpoint",
    "load_checkpoint",
    "ExperimentLogger",
    "Trainer",
    "Evaluator",
    "profile_model",
    "ExperimentRunner",
    "ABLATION_VARIANTS",
    "NoiseScheduler",
    "DiffusionAuxiliary",
]
