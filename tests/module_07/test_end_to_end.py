"""End-to-End Integration Tests for Module 7.

Pipeline:
  Synthetic Scene Data (Modules 1-3)
    ↓
  Module 4 (FP32 Mamba-Hybrid Engine) & Module 6 (BitNet Engine)
    ↓
  pooled_output [B, D_model]
    ↓
  Module 7 PhotonShieldMultiTask (Target, Anomaly, Environmental Heads)
    ↓
  Module 7 DecisionLogic Layer (Probabilities, Hysteresis, Causal Smoothing)
    ↓
  Structured Application Decision Outputs
"""

import pytest
import torch
import tempfile
from torch.utils.data import DataLoader

from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_05_training.dataset import make_synthetic_scene_cache
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.layer_replacement import replace_linear_layers
from module_07_decision.config import DecisionModelConfig, DecisionConfig
from module_07_decision.multitask import PhotonShieldMultiTask, MultiTaskDecisionLoss
from module_07_decision.decision_logic import DecisionLogic
from module_07_decision.inference import PhotonShieldDecisionPipeline


class TestModule07EndToEnd:
    def test_fp32_mamba_to_module07_pipeline(self):
        """Verify FP32 Module 4 engine integration with Module 7."""
        # 1. Module 4 engine
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        engine_fp32 = PhotonMambaHybrid(m_cfg)

        # 2. Module 7 MultiTask & Decision Logic
        d_m_cfg = DecisionModelConfig(d_model=32, enable_target=True, enable_anomaly=True, enable_environment=True)
        d_cfg = DecisionConfig()
        multi_task = PhotonShieldMultiTask(d_m_cfg)
        pipeline = PhotonShieldDecisionPipeline(multi_task, d_cfg, engine=engine_fp32)

        # 3. Dummy fused sample batch
        cache = make_synthetic_scene_cache(num_scenes=1, frames_per_scene=10)
        window = cache.get_window("synthetic_scene_000", 0, 10)
        batch = {k: v.unsqueeze(0) for k, v in window.items() if isinstance(v, torch.Tensor)}

        # 4. Predict
        decisions = pipeline.predict_sample(batch)
        assert len(decisions) == 1
        assert decisions[0].combined_event_state in ["NORMAL", "TARGET", "ANOMALY", "TARGET_AND_ANOMALY"]

    def test_bitnet_mamba_to_module07_pipeline(self):
        """Verify BitNet Module 4 engine integration with Module 7."""
        # 1. Module 4 engine converted to BitNet
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        engine = PhotonMambaHybrid(m_cfg)
        b_cfg = BitNetConfig()
        replace_linear_layers(engine, b_cfg)

        # 2. Module 7 MultiTask & Decision Logic
        d_m_cfg = DecisionModelConfig(d_model=32)
        d_cfg = DecisionConfig()
        multi_task = PhotonShieldMultiTask(d_m_cfg)
        pipeline = PhotonShieldDecisionPipeline(multi_task, d_cfg, engine=engine)

        # 3. Dummy fused sample batch
        cache = make_synthetic_scene_cache(num_scenes=1, frames_per_scene=10)
        window = cache.get_window("synthetic_scene_000", 0, 10)
        batch = {k: v.unsqueeze(0) for k, v in window.items() if isinstance(v, torch.Tensor)}

        # 4. Predict
        decisions = pipeline.predict_sample(batch)
        assert len(decisions) == 1
        assert torch.isfinite(torch.tensor(decisions[0].target_probability))
