"""High-level ExperimentRunner for Module 5.

Provides a single entry point for running the baseline experiment and
ablation variants. Generates a machine-readable JSON summary report.

NOTE: This module does NOT automatically run expensive experiments.
Experiments are triggered only when explicitly called.
"""

from __future__ import annotations
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from module_05_training.config import TrainingConfig
from module_05_training.dataset import (
    PhotonShieldDataset,
    SceneFeatureCache,
    collate_module3,
)
from module_05_training.target_adapter import get_target_adapter
from module_05_training.trainer import Trainer
from module_05_training.evaluator import Evaluator
from module_05_training.profiling import profile_model
from module_05_training.reproducibility import set_seed


# ── Ablation variant configurations ───────────────────────────────────────────
ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline_mamba": {
        "use_mamba": True,
        "use_sensor_attention": False,
    },
    "sensor_interaction_only": {
        "use_mamba": False,
        "use_sensor_attention": True,
    },
    "full_mamba_hybrid": {
        "use_mamba": True,
        "use_sensor_attention": True,
    },
}


class ExperimentRunner:
    """Orchestrates training, evaluation, and reporting for one experiment.

    Args:
        train_config: TrainingConfig specifying training hyperparameters.
        train_cache: SceneFeatureCache for the training split.
        val_cache: SceneFeatureCache for the validation split.
        test_cache: SceneFeatureCache for the test split.
        model_config: MambaHybridConfig instance.
        report_dir: Directory where the JSON report will be written.
    """

    def __init__(
        self,
        train_config: TrainingConfig,
        train_cache: SceneFeatureCache,
        val_cache: SceneFeatureCache,
        test_cache: SceneFeatureCache,
        model_config: Any,
        report_dir: str = "reports",
    ):
        self.train_config = train_config
        self.train_cache = train_cache
        self.val_cache = val_cache
        self.test_cache = test_cache
        self.model_config = model_config
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _make_loaders(self) -> Dict[str, DataLoader]:
        adapter = get_target_adapter(self.train_config)
        cfg = self.train_config

        train_ds = PhotonShieldDataset(
            self.train_cache, adapter,
            window_len=cfg.sequence_length,
            window_stride=cfg.sequence_stride,
        )
        val_ds = PhotonShieldDataset(
            self.val_cache, adapter,
            window_len=cfg.sequence_length,
            window_stride=cfg.sequence_stride,
        )
        test_ds = PhotonShieldDataset(
            self.test_cache, adapter,
            window_len=cfg.sequence_length,
            window_stride=cfg.sequence_stride,
        )

        loaders = {
            "train": DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=True,
                num_workers=cfg.num_workers, collate_fn=collate_module3,
            ),
            "val": DataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, collate_fn=collate_module3,
            ),
            "test": DataLoader(
                test_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, collate_fn=collate_module3,
            ),
        }
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val   samples: {len(val_ds)}")
        print(f"  Test  samples: {len(test_ds)}")
        return loaders, {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        }

    def _build_model(self, variant: Optional[str] = None):
        """Build engine + head from model_config with optional variant overrides."""
        from module_04_mamba_hybrid.config import MambaHybridConfig
        from module_04_mamba_hybrid.engine import PhotonMambaHybrid
        from module_04_mamba_hybrid.heads import ClassificationHead, RegressionHead

        cfg = self.model_config
        if variant and variant in ABLATION_VARIANTS:
            overrides = ABLATION_VARIANTS[variant]
            from dataclasses import replace
            cfg = replace(cfg, **overrides)

        engine = PhotonMambaHybrid(cfg)

        from module_04_mamba_hybrid.config import TaskHeadConfig
        if self.train_config.target_type == "classification":
            head_cfg = TaskHeadConfig(head_type="classification", num_classes=self.train_config.num_classes)
            head = ClassificationHead(cfg.d_model, head_cfg)
        else:
            head_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=self.train_config.num_regression_outputs)
            head = RegressionHead(cfg.d_model, head_cfg)

        return engine, head

    def run(self, variant: str = "full_mamba_hybrid") -> Dict[str, Any]:
        """Run a full training + evaluation experiment.

        Args:
            variant: Ablation variant name. Default: "full_mamba_hybrid".

        Returns:
            JSON-serialisable summary dict.
        """
        print(f"\n[Experiment] Running variant: {variant}")
        set_seed(self.train_config.random_seed)

        t_run_start = time.time()
        loaders, split_sizes = self._make_loaders()
        engine, head = self._build_model(variant)

        trainer = Trainer(
            engine=engine,
            head=head,
            config=self.train_config,
            model_config=self.model_config,
            experiment_name=f"photonshield_{variant}",
        )

        train_summary = trainer.fit(loaders["train"], loaders["val"])

        # Test evaluation (ONLY after training is complete)
        print("\n[Experiment] Running test evaluation...")
        evaluator = Evaluator(self.train_config)
        test_results = evaluator.evaluate(engine, head, loaders["test"])

        # Profiling (single sample)
        print("[Experiment] Profiling...")
        sample_window, _ = next(iter(loaders["test"]))
        # Unbatch first sample
        single = {
            k: (v[0] if isinstance(v, torch.Tensor) else v)
            for k, v in sample_window.items()
        }
        device = next(engine.parameters()).device
        prof = profile_model(engine, head, single, device)

        total_time_s = time.time() - t_run_start

        report = {
            "variant": variant,
            "model_config": (
                asdict(self.model_config) if is_dataclass(self.model_config) else {}
            ),
            "training_config": (
                asdict(self.train_config) if is_dataclass(self.train_config) else {}
            ),
            "split_sizes": split_sizes,
            "training_summary": train_summary,
            "test_results": test_results,
            "profiling": prof,
            "total_experiment_time_s": round(total_time_s, 2),
        }

        report_path = self.report_dir / f"report_{variant}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[Experiment] Report saved: {report_path}")

        return report
