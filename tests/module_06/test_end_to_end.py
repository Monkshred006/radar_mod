"""End-to-End Integration Tests for Module 6.

Pipeline:
  Synthetic Scene Data
    ↓
  Module 1 (Scene Splits)
    ↓
  Module 2 (Offline DSP)
    ↓
  Module 3 (Offline Fusion)
    ↓
  Module 4 (FP32 Mamba-Hybrid Engine)
    ↓
  Module 5 (FP32 Trainer → FP32 Reference Checkpoint)
    ↓
  Module 6 (Model Conversion → Direct-Ternary PTQ → BitNet QAT Fine-tuning)
    ↓
  Module 5 Evaluator (Fair Evaluation on identical test set)
    ↓
  Comparison Matrix Report
"""

import pytest
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import RegressionHead
from module_05_training.config import TrainingConfig
from module_05_training.dataset import make_synthetic_scene_cache, PhotonShieldDataset, collate_module3
from module_05_training.target_adapter import SyntheticRegressionAdapter
from module_05_training.trainer import Trainer
from module_05_training.checkpointing import save_checkpoint
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.experiment import BitNetExperimentRunner


class TestModule06EndToEnd:
    def test_full_module06_pipeline(self, tmp_path):
        # 1. Setup small synthetic datasets & caches
        train_cache = make_synthetic_scene_cache(num_scenes=3, frames_per_scene=30)
        val_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=20)
        test_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=20)

        # 2. Build and train reference FP32 model (Module 5)
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        t_cfg = TrainingConfig(
            epochs=1,
            learning_rate=1e-3,
            batch_size=4,
            device="cpu",
            num_regression_outputs=1,
            checkpoint_dir=str(tmp_path / "fp32_ckpts"),
            log_dir=str(tmp_path / "fp32_logs"),
        )
        engine_fp32 = PhotonMambaHybrid(m_cfg)
        h_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=1)
        head_fp32 = RegressionHead(32, h_cfg)

        trainer_fp32 = Trainer(engine_fp32, head_fp32, t_cfg, model_config=m_cfg)

        adapter = SyntheticRegressionAdapter(num_outputs=1)
        train_ds = PhotonShieldDataset(train_cache, adapter, window_len=10, window_stride=5)
        val_ds = PhotonShieldDataset(val_cache, adapter, window_len=10, window_stride=5)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate_module3)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_module3)

        trainer_fp32.fit(train_loader, val_loader)

        fp32_ckpt_path = str(tmp_path / "fp32_ckpts" / "photonshield_m5_latest.pt")
        if not Path(fp32_ckpt_path).exists():
            save_checkpoint(
                path=str(tmp_path / "manual_fp32.pt"),
                model=torch.nn.ModuleList([engine_fp32, head_fp32]),
                optimizer=trainer_fp32.optimizer,
                scheduler=None,
                epoch=1,
                best_val_metric=0.5,
                training_config=t_cfg,
                model_config=m_cfg,
                history=[],
                seed=42,
            )
            fp32_ckpt_path = str(tmp_path / "manual_fp32.pt")

        # 3. Execute Module 6 BitNet experiment runner
        b_cfg = BitNetConfig(epochs=1, learning_rate=1e-3)
        runner = BitNetExperimentRunner(
            fp32_checkpoint_path=fp32_ckpt_path,
            train_config=t_cfg,
            train_cache=train_cache,
            val_cache=val_cache,
            test_cache=test_cache,
            bitnet_config=b_cfg,
            output_dir=str(tmp_path / "bitnet_reports"),
        )

        matrix_report = runner.run_comparison_experiment()

        # 4. Verify comparison outputs
        assert "comparison_rows" in matrix_report
        assert len(matrix_report["comparison_rows"]) == 3
        assert "markdown_table" in matrix_report
        assert (tmp_path / "bitnet_reports" / "comparison_matrix.json").exists()
        assert (tmp_path / "bitnet_reports" / "comparison_matrix.md").exists()
