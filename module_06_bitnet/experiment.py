"""BitNet Experiment Runner & Ablation Suite for Module 6.

Orchestrates ablation experiments across quantization configurations:
  Experiment A: FP32 Baseline
  Experiment B: Input projection ternary only
  Experiment C: Attention Q/K/V/O ternary only
  Experiment D: FFN ternary only
  Experiment E: Input + Attention ternary
  Experiment F: Attention + FFN ternary
  Experiment G: Full supported BitNet QAT
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader

from module_05_training.config import TrainingConfig
from module_05_training.dataset import SceneFeatureCache, PhotonShieldDataset, collate_module3
from module_05_training.target_adapter import get_target_adapter
from module_05_training.evaluator import Evaluator
from module_05_training.reproducibility import set_seed
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.model_conversion import convert_fp32_to_bitnet
from module_06_bitnet.pta_baseline import evaluate_ptq_baseline
from module_06_bitnet.qat import BitNetQATTrainer
from module_06_bitnet.profiling import profile_bitnet_model
from module_06_bitnet.comparison import generate_comparison_matrix


BITNET_ABLATION_CONFIGS: Dict[str, Dict[str, bool]] = {
    "exp_a_fp32_baseline": {
        "quantize_input_projection": False,
        "quantize_sensor_attention_qkv": False,
        "quantize_sensor_attention_output": False,
        "quantize_ffn": False,
        "quantize_task_head": False,
    },
    "exp_b_input_only": {
        "quantize_input_projection": True,
        "quantize_sensor_attention_qkv": False,
        "quantize_sensor_attention_output": False,
        "quantize_ffn": False,
        "quantize_task_head": False,
    },
    "exp_c_attention_only": {
        "quantize_input_projection": False,
        "quantize_sensor_attention_qkv": True,
        "quantize_sensor_attention_output": True,
        "quantize_ffn": False,
        "quantize_task_head": False,
    },
    "exp_d_ffn_only": {
        "quantize_input_projection": False,
        "quantize_sensor_attention_qkv": False,
        "quantize_sensor_attention_output": False,
        "quantize_ffn": True,
        "quantize_task_head": False,
    },
    "exp_e_input_attention": {
        "quantize_input_projection": True,
        "quantize_sensor_attention_qkv": True,
        "quantize_sensor_attention_output": True,
        "quantize_ffn": False,
        "quantize_task_head": False,
    },
    "exp_f_attention_ffn": {
        "quantize_input_projection": False,
        "quantize_sensor_attention_qkv": True,
        "quantize_sensor_attention_output": True,
        "quantize_ffn": True,
        "quantize_task_head": False,
    },
    "exp_g_full_bitnet": {
        "quantize_input_projection": True,
        "quantize_sensor_attention_qkv": True,
        "quantize_sensor_attention_output": True,
        "quantize_ffn": True,
        "quantize_task_head": False,
    },
}


class BitNetExperimentRunner:
    """Orchestrates BitNet conversion, Direct-Ternary PTQ, QAT fine-tuning, and profiling.

    Args:
        fp32_checkpoint_path: Path to reference FP32 checkpoint file.
        train_config: TrainingConfig specifying loss, device, etc.
        train_cache: Training split SceneFeatureCache.
        val_cache: Validation split SceneFeatureCache.
        test_cache: Test split SceneFeatureCache.
        bitnet_config: BitNetConfig.
        output_dir: Output directory for reports and checkpoints.
    """

    def __init__(
        self,
        fp32_checkpoint_path: str,
        train_config: TrainingConfig,
        train_cache: SceneFeatureCache,
        val_cache: SceneFeatureCache,
        test_cache: SceneFeatureCache,
        bitnet_config: Optional[BitNetConfig] = None,
        output_dir: str = "reports/bitnet",
    ):
        self.fp32_checkpoint_path = fp32_checkpoint_path
        self.train_config = train_config
        self.train_cache = train_cache
        self.val_cache = val_cache
        self.test_cache = test_cache
        self.bitnet_config = bitnet_config or BitNetConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_loaders(self) -> Dict[str, DataLoader]:
        adapter = get_target_adapter(self.train_config)
        cfg = self.train_config

        train_ds = PhotonShieldDataset(self.train_cache, adapter, window_len=cfg.sequence_length, window_stride=cfg.sequence_stride)
        val_ds = PhotonShieldDataset(self.val_cache, adapter, window_len=cfg.sequence_length, window_stride=cfg.sequence_stride)
        test_ds = PhotonShieldDataset(self.test_cache, adapter, window_len=cfg.sequence_length, window_stride=cfg.sequence_stride)

        return {
            "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_module3),
            "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_module3),
            "test": DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_module3),
        }

    def run_comparison_experiment(self) -> Dict[str, Any]:
        """Run standard comparison: FP32 Baseline vs Direct-Ternary PTQ vs BitNet-Style QAT."""
        set_seed(self.train_config.random_seed)
        loaders = self._make_loaders()
        sample_batch, _ = next(iter(loaders["test"]))
        single_sample = {k: (v[0] if isinstance(v, torch.Tensor) else v) for k, v in sample_batch.items()}

        # 1. FP32 Evaluation & Profiling
        print("\n[BitNet Experiment] Evaluating FP32 Baseline...")
        from module_05_training.evaluator import Evaluator
        from module_04_mamba_hybrid.engine import PhotonMambaHybrid
        from module_04_mamba_hybrid.heads import ClassificationHead, RegressionHead
        from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig

        raw_fp32 = torch.load(self.fp32_checkpoint_path, map_location="cpu", weights_only=False)
        m_cfg = MambaHybridConfig(**raw_fp32["model_config"])
        fp32_engine = PhotonMambaHybrid(m_cfg)
        if self.train_config.target_type == "classification":
            h_cfg = TaskHeadConfig(head_type="classification", num_classes=self.train_config.num_classes)
            fp32_head = ClassificationHead(m_cfg.d_model, h_cfg)
        else:
            h_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=self.train_config.num_regression_outputs)
            fp32_head = RegressionHead(m_cfg.d_model, h_cfg)

        combined_fp32 = torch.nn.ModuleList([fp32_engine, fp32_head])
        combined_fp32.load_state_dict(raw_fp32["model_state_dict"])

        evaluator = Evaluator(self.train_config)
        fp32_eval = evaluator.evaluate(fp32_engine, fp32_head, loaders["test"])
        fp32_prof = profile_bitnet_model(fp32_engine, fp32_head, single_sample, self.fp32_checkpoint_path)

        # 2. Conversion to BitNet & Direct-Ternary PTQ Evaluation
        print("[BitNet Experiment] Converting to BitNet and evaluating Direct-Ternary PTQ...")
        bitnet_ckpt_path = str(self.output_dir / "bitnet_ptq.pt")
        ptq_engine, ptq_head, conversion_report = convert_fp32_to_bitnet(
            self.fp32_checkpoint_path, self.bitnet_config, bitnet_ckpt_path
        )
        ptq_eval = evaluate_ptq_baseline(ptq_engine, ptq_head, self.train_config, loaders["test"])
        ptq_prof = profile_bitnet_model(ptq_engine, ptq_head, single_sample, bitnet_ckpt_path)

        # 3. BitNet QAT Fine-Tuning
        print("[BitNet Experiment] Running BitNet-Style QAT Fine-Tuning...")
        qat_trainer = BitNetQATTrainer(
            engine=ptq_engine,
            head=ptq_head,
            bitnet_config=self.bitnet_config,
            train_config=self.train_config,
            model_config=m_cfg,
            source_fp32_checkpoint=self.fp32_checkpoint_path,
        )
        qat_summary = qat_trainer.train_qat(loaders["train"], loaders["val"])
        qat_ckpt_path = qat_summary["checkpoint_path"]

        qat_eval = evaluator.evaluate(ptq_engine, ptq_head, loaders["test"])
        qat_prof = profile_bitnet_model(ptq_engine, ptq_head, single_sample, qat_ckpt_path)

        # 4. Generate Comparison Matrix
        print("[BitNet Experiment] Generating Comparison Matrix Report...")
        matrix_report = generate_comparison_matrix(
            fp32_eval_results=fp32_eval,
            fp32_prof=fp32_prof,
            ptq_eval_results=ptq_eval,
            ptq_prof=ptq_prof,
            qat_eval_results=qat_eval,
            qat_prof=qat_prof,
            output_dir=str(self.output_dir),
        )

        matrix_report["conversion_report"] = conversion_report
        return matrix_report
