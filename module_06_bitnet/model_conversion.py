"""FP32 to BitNet-Compatible Model Conversion Pipeline.

Converts a trained FP32 Module 4 engine + task head into a BitNet-compatible
mixed-precision model using selective layer replacement.
Guarantees FP32 reference checkpoint immutability.
Calculates layer-wise conversion statistics and quantization error metrics.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.heads import ClassificationHead, RegressionHead
from module_05_training.config import TrainingConfig
from module_05_training.checkpointing import load_checkpoint
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.bit_linear import BitLinear
from module_06_bitnet.layer_replacement import replace_linear_layers, generate_layer_inspection_report
from module_06_bitnet.ternary import round_to_ternary
from module_06_bitnet.checkpointing import save_bitnet_checkpoint


def _file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file for immutability verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_layer_conversion_stats(bitnet_model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Compute layer-wise weight statistics and quantization error metrics.

    Returns:
        Dict mapping layer_name -> {
            fp32_mean, fp32_std, fp32_min, fp32_max,
            scale_alpha, pct_minus_1, pct_0, pct_plus_1,
            quant_mse_error, quant_mae_error
        }
    """
    stats: Dict[str, Dict[str, float]] = {}

    for name, module in bitnet_model.named_modules():
        if isinstance(module, BitLinear):
            w = module.weight.data
            w_quant, scale, w_int = round_to_ternary(
                w,
                scale_method=module.config.scaling_method,
                scale_scope=module.config.scaling_scope,
            )

            # Error metrics between FP32 master weight and scaled ternary weight
            diff = w - w_quant
            mse = torch.mean(diff ** 2).item()
            mae = torch.mean(torch.abs(diff)).item()

            # Symbol distribution
            n = float(w_int.numel())
            cnt_minus_1 = (w_int == -1).sum().item()
            cnt_0 = (w_int == 0).sum().item()
            cnt_plus_1 = (w_int == 1).sum().item()

            scale_val = scale.mean().item() if scale.ndim > 0 else scale.item()

            stats[name] = {
                "fp32_mean": float(w.mean().item()),
                "fp32_std": float(w.std().item()),
                "fp32_min": float(w.min().item()),
                "fp32_max": float(w.max().item()),
                "scale_alpha": scale_val,
                "pct_minus_1": round(cnt_minus_1 / n * 100.0, 2),
                "pct_0": round(cnt_0 / n * 100.0, 2),
                "pct_plus_1": round(cnt_plus_1 / n * 100.0, 2),
                "quant_mse_error": mse,
                "quant_mae_error": mae,
            }

    return stats


def convert_fp32_to_bitnet(
    fp32_checkpoint_path: str,
    bitnet_config: BitNetConfig,
    output_checkpoint_path: str,
) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
    """Convert an FP32 Module 5 model into a BitNet-compatible model.

    Args:
        fp32_checkpoint_path: Path to existing FP32 checkpoint file (.pt).
        bitnet_config: BitNetConfig specifying quantization policy.
        output_checkpoint_path: Destination path for the converted BitNet checkpoint.

    Returns:
        Tuple of (bitnet_engine, bitnet_head, conversion_report_dict).
    """
    ckpt_path = Path(fp32_checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"FP32 checkpoint not found: {ckpt_path}")

    # 1. Hash FP32 checkpoint before conversion to guarantee immutability
    hash_before = _file_sha256(str(ckpt_path))

    # 2. Load raw checkpoint data
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Reconstruct configs
    train_cfg_dict = raw_ckpt.get("training_config", {})
    model_cfg_dict = raw_ckpt.get("model_config", {})

    train_cfg = TrainingConfig(**{
        k: v for k, v in train_cfg_dict.items()
        if k in TrainingConfig.__dataclass_fields__
    })
    model_cfg = MambaHybridConfig(**{
        k: v for k, v in model_cfg_dict.items()
        if k in MambaHybridConfig.__dataclass_fields__
    })

    # 3. Rebuild FP32 engine + head
    engine_fp32 = PhotonMambaHybrid(model_cfg)
    if train_cfg.target_type == "classification":
        head_cfg = TaskHeadConfig(head_type="classification", num_classes=train_cfg.num_classes)
        head_fp32 = ClassificationHead(model_cfg.d_model, head_cfg)
    else:
        head_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=train_cfg.num_regression_outputs)
        head_fp32 = RegressionHead(model_cfg.d_model, head_cfg)

    combined_fp32 = nn.ModuleList([engine_fp32, head_fp32])
    load_checkpoint(str(ckpt_path), combined_fp32)

    # 4. Perform selective layer replacement to build BitNet models
    engine_bitnet, engine_summary = replace_linear_layers(engine_fp32, bitnet_config)
    head_bitnet, head_summary = replace_linear_layers(head_fp32, bitnet_config)

    # 5. Generate layer inspection report & conversion stats
    combined_bitnet = nn.ModuleList([engine_bitnet, head_bitnet])
    inspection_report = generate_layer_inspection_report(combined_bitnet, bitnet_config)
    layer_stats = compute_layer_conversion_stats(combined_bitnet)

    # 6. Verify FP32 checkpoint immutability
    hash_after = _file_sha256(str(ckpt_path))
    if hash_before != hash_after:
        raise RuntimeError("CRITICAL ERROR: FP32 source checkpoint was modified during conversion!")

    # 7. Save new BitNet checkpoint (in separate output directory)
    save_bitnet_checkpoint(
        path=output_checkpoint_path,
        engine=engine_bitnet,
        head=head_bitnet,
        bitnet_config=bitnet_config,
        model_config=model_cfg,
        training_config=train_cfg,
        source_fp32_checkpoint=str(ckpt_path),
        layer_stats=layer_stats,
    )

    report = {
        "source_fp32_checkpoint": str(ckpt_path),
        "output_bitnet_checkpoint": output_checkpoint_path,
        "source_hash": hash_before,
        "engine_replacement_summary": engine_summary,
        "head_replacement_summary": head_summary,
        "layer_inspection_report": inspection_report,
        "layer_conversion_stats": layer_stats,
    }

    return engine_bitnet, head_bitnet, report
