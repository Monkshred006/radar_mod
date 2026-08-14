"""Comparison Matrix Generation Utilities for Module 6.

Compares:
  1. FP32 Baseline
  2. Direct-Ternary PTQ
  3. BitNet-Style QAT

Generates machine-readable JSON and formatted Markdown comparison tables.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def generate_comparison_matrix(
    fp32_eval_results: Dict[str, Any],
    fp32_prof: Dict[str, Any],
    ptq_eval_results: Dict[str, Any],
    ptq_prof: Dict[str, Any],
    qat_eval_results: Dict[str, Any],
    qat_prof: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate structured comparison report matrix for FP32 vs PTQ vs BitNet QAT.

    Args:
        fp32_eval_results: Evaluation results dict for FP32 model.
        fp32_prof: Profiling dict for FP32 model.
        ptq_eval_results: Evaluation results dict for Direct-Ternary PTQ model.
        ptq_prof: Profiling dict for Direct-Ternary PTQ model.
        qat_eval_results: Evaluation results dict for BitNet-Style QAT model.
        qat_prof: Profiling dict for BitNet-Style QAT model.
        output_dir: Optional directory path to write JSON and Markdown reports.

    Returns:
        Structured comparison matrix dictionary.
    """
    rows: List[Dict[str, Any]] = [
        _build_comparison_row("FP32 Baseline", fp32_eval_results, fp32_prof),
        _build_comparison_row("Direct-Ternary PTQ", ptq_eval_results, ptq_prof),
        _build_comparison_row("BitNet-Style QAT", qat_eval_results, qat_prof),
    ]

    markdown_table = _build_markdown_table(rows)

    matrix_report = {
        "comparison_rows": rows,
        "markdown_table": markdown_table,
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "comparison_matrix.json", "w", encoding="utf-8") as f:
            json.dump(matrix_report, f, indent=2, default=str)
        with open(out_path / "comparison_matrix.md", "w", encoding="utf-8") as f:
            f.write(markdown_table)

    return matrix_report


def _build_comparison_row(
    label: str,
    eval_results: Dict[str, Any],
    prof: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = eval_results.get("metrics", {})
    primary_metric_val = (
        metrics.get("mae") or metrics.get("accuracy") or eval_results.get("loss", float("nan"))
    )

    return {
        "variant": label,
        "quantized_layers": f"{prof.get('pct_ternary', 0.0)}%",
        "ternary_param_pct": prof.get("pct_ternary", 0.0),
        "fp32_param_pct": prof.get("pct_fp32", 100.0),
        "theoretical_bits_per_weight": prof.get("theoretical_bits_per_weight", 32.0),
        "actual_checkpoint_mb": prof.get("actual_checkpoint_mb", 0.0),
        "loss": eval_results.get("loss", float("nan")),
        "primary_metric": round(primary_metric_val, 5) if isinstance(primary_metric_val, (int, float)) else primary_metric_val,
        "mean_latency_ms": prof.get("mean_latency_ms", float("nan")),
        "p50_latency_ms": prof.get("p50_latency_ms", float("nan")),
        "p95_latency_ms": prof.get("p95_latency_ms", float("nan")),
        "throughput_samples_per_sec": prof.get("throughput_samples_per_sec", 0.0),
        "peak_gpu_mem_mb": prof.get("peak_gpu_mem_mb", 0.0),
        "device": prof.get("device", "CPU"),
        "hardware_disclaimer": prof.get("hardware_disclaimer", ""),
    }


def _build_markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Model Variant", "Ternary %", "Bits/Weight (Theoretical)", "Checkpoint (MB)",
        "Loss", "Primary Metric", "Mean Latency (ms)", "Throughput (smp/s)", "Device"
    ]
    header_str = " | ".join(headers)
    divider_str = " | ".join(["---"] * len(headers))
    lines = [f"| {header_str} |", f"| {divider_str} |"]

    for r in rows:
        row_str = (
            f"| {r['variant']} | {r['ternary_param_pct']}% | {r['theoretical_bits_per_weight']} | "
            f"{r['actual_checkpoint_mb']} | {r['loss']:.4f} | {r['primary_metric']} | "
            f"{r['mean_latency_ms']} | {r['throughput_samples_per_sec']} | {r['device']} |"
        )
        lines.append(row_str)

    lines.append("\n> **Hardware Disclaimer**: Low-bit model validated at the neural-model level; dedicated ternary hardware acceleration has not yet been validated.")
    return "\n".join(lines)
