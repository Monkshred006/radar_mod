"""Selective Layer Replacement & Layer Inspection Utilities for Module 6.

Scans PyTorch models and selectively converts nn.Linear layers into BitLinear
based on BitNetConfig granular quantization flags.
Preserves sensitive operations (Mamba state-space core, LayerNorm, activations).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn

from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.bit_linear import BitLinear


def _should_quantize_layer(name: str, module: nn.Module, config: BitNetConfig) -> bool:
    """Determine whether a specific nn.Linear layer should be converted to BitLinear.

    Args:
        name: Full attribute path of the layer (e.g. 'input_projection.token_proj').
        module: The nn.Module instance.
        config: BitNetConfig with layer selection flags.

    Returns:
        Boolean indicating if the layer should be converted.
    """
    if not isinstance(module, nn.Linear):
        return False

    name_lower = name.lower()

    # Mamba internal core layers
    if "mamba_branch" in name_lower or "ssm" in name_lower:
        return config.quantize_mamba_internal

    # Input projection layers
    if "input_projection" in name_lower or "token_proj" in name_lower or "flat_proj" in name_lower:
        return config.quantize_input_projection

    # Sensor Interaction Attention Q/K/V projections
    if any(qkv in name_lower for qkv in ("q_proj", "k_proj", "v_proj")):
        return config.quantize_sensor_attention_qkv

    # Sensor Interaction Attention Output projection
    if "out_proj" in name_lower:
        return config.quantize_sensor_attention_output

    # FFN projections
    if "ffn" in name_lower or "fc1" in name_lower or "fc2" in name_lower or "net" in name_lower:
        # Check if it's inside task head
        if "head" in name_lower:
            return config.quantize_task_head
        return config.quantize_ffn

    # Task head layers
    if "head" in name_lower:
        return config.quantize_task_head

    # Default fallback for any remaining nn.Linear
    return False


def replace_linear_layers(
    model: nn.Module,
    config: BitNetConfig,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Recursively traverse model and replace selected nn.Linear layers with BitLinear.

    Args:
        model: PyTorch model instance (e.g. PhotonMambaHybrid or task head).
        config: BitNetConfig specifying layer selection policy.

    Returns:
        Tuple of (model, conversion_summary_dict).
    """
    replaced_count = 0
    preserved_count = 0
    replacement_map: Dict[str, str] = {}

    # Gather targets for replacement to avoid mutating dictionary during iteration
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if _should_quantize_layer(name, module, config):
                # Find parent module and attribute name
                *parent_path, attr_name = name.split(".")
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)

                # Convert to BitLinear
                bit_layer = BitLinear.from_linear(module, config=config)
                setattr(parent, attr_name, bit_layer)

                replaced_count += 1
                replacement_map[name] = "BitLinear (ternary)"
            else:
                preserved_count += 1
                replacement_map[name] = "nn.Linear (FP32 preserved)"

    summary = {
        "replaced_count": replaced_count,
        "preserved_count": preserved_count,
        "total_linear_layers": replaced_count + preserved_count,
        "replacement_map": replacement_map,
    }

    return model, summary


def generate_layer_inspection_report(
    model: nn.Module,
    config: BitNetConfig,
) -> Dict[str, Any]:
    """Generate an automated Layer Inspection Report for a model.

    Args:
        model: Model to inspect.
        config: Active BitNetConfig.

    Returns:
        Dict containing structured layer data and a formatted ASCII table string.
    """
    layers_data: List[Dict[str, Any]] = []

    total_params = 0
    ternary_params = 0
    fp32_params = 0

    for name, module in model.named_modules():
        # Only inspect leaf parameter-containing layers
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if n_params == 0 and not isinstance(module, (nn.Linear, BitLinear)):
            continue

        if isinstance(module, BitLinear):
            status = "YES"
            prec = f"ternary ({config.activation_precision})"
            scaling = f"{config.scaling_method}/{config.scaling_scope}"
            ternary_params += n_params
            total_params += n_params
            layers_data.append({
                "layer_name": name,
                "layer_type": "BitLinear",
                "param_count": n_params,
                "quantized": status,
                "precision": prec,
                "scaling": scaling,
            })
        elif isinstance(module, nn.Linear):
            status = "NO"
            prec = "FP32"
            scaling = "N/A"
            fp32_params += n_params
            total_params += n_params
            layers_data.append({
                "layer_name": name,
                "layer_type": "nn.Linear",
                "param_count": n_params,
                "quantized": status,
                "precision": prec,
                "scaling": scaling,
            })

    pct_ternary = (ternary_params / total_params * 100.0) if total_params > 0 else 0.0

    # Build ASCII report table
    header = f"{'Layer Name':<40} {'Type':<12} {'Params':<10} {'Quantized':<10} {'Precision':<20}"
    divider = "-" * 95
    lines = [divider, header, divider]
    for row in layers_data:
        lines.append(
            f"{row['layer_name']:<40} {row['layer_type']:<12} {row['param_count']:<10,d} "
            f"{row['quantized']:<10} {row['precision']:<20}"
        )
    lines.append(divider)
    lines.append(f"Total Parameters:      {total_params:,d}")
    lines.append(f"Ternary Parameters:    {ternary_params:,d} ({pct_ternary:.1f}%)")
    lines.append(f"FP32 Parameters:       {fp32_params:,d} ({100.0 - pct_ternary:.1f}%)")
    lines.append(divider)

    formatted_table = "\n".join(lines)

    return {
        "layers": layers_data,
        "total_params": total_params,
        "ternary_params": ternary_params,
        "fp32_params": fp32_params,
        "pct_ternary": round(pct_ternary, 2),
        "formatted_table": formatted_table,
    }
