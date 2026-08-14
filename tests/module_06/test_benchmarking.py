"""Tests for profiling and comparison matrix benchmarking utilities."""

import pytest
import torch
from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import RegressionHead
from module_05_training.dataset import make_synthetic_scene_cache
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.layer_replacement import replace_linear_layers
from module_06_bitnet.profiling import profile_bitnet_model
from module_06_bitnet.comparison import generate_comparison_matrix


class TestBenchmarking:
    def test_profile_bitnet_model_keys(self):
        m_cfg = MambaHybridConfig(d_model=32, num_layers=1, max_sequence_length=10)
        engine = PhotonMambaHybrid(m_cfg)
        h_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=1)
        head = RegressionHead(32, h_cfg)

        b_cfg = BitNetConfig()
        replace_linear_layers(engine, b_cfg)

        cache = make_synthetic_scene_cache(num_scenes=1, frames_per_scene=20)
        window = cache.get_window("synthetic_scene_000", 0, 10)

        prof = profile_bitnet_model(engine, head, window, n_warmup=1, n_runs=2)

        assert "total_params" in prof
        assert "ternary_params" in prof
        assert "pct_ternary" in prof
        assert "theoretical_bits_per_weight" in prof
        assert "mean_latency_ms" in prof
        assert "hardware_disclaimer" in prof

    def test_comparison_matrix_generation(self):
        eval_dummy = {"loss": 0.1, "metrics": {"mae": 0.05}}
        prof_dummy = {
            "pct_ternary": 80.0,
            "pct_fp32": 20.0,
            "theoretical_bits_per_weight": 7.7,
            "actual_checkpoint_mb": 1.2,
            "mean_latency_ms": 5.0,
            "throughput_samples_per_sec": 200.0,
            "device": "CPU",
            "hardware_disclaimer": "test disclaimer",
        }

        matrix = generate_comparison_matrix(
            fp32_eval_results=eval_dummy,
            fp32_prof={**prof_dummy, "pct_ternary": 0.0, "pct_fp32": 100.0},
            ptq_eval_results=eval_dummy,
            ptq_prof=prof_dummy,
            qat_eval_results=eval_dummy,
            qat_prof=prof_dummy,
        )

        assert "comparison_rows" in matrix
        assert len(matrix["comparison_rows"]) == 3
        assert "markdown_table" in matrix
        assert "| FP32 Baseline |" in matrix["markdown_table"]
        assert "| Direct-Ternary PTQ |" in matrix["markdown_table"]
        assert "| BitNet-Style QAT |" in matrix["markdown_table"]
