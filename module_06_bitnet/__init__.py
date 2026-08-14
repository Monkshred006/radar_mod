"""Module 6: BitNet / 1.58-Bit Model Optimization & Benchmarking for PhotonShield AI."""

from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.scaling import compute_weight_scale, compute_activation_scale
from module_06_bitnet.ternary import (
    TernarySTEFunction,
    round_to_ternary,
    pack_ternary,
    unpack_ternary,
)
from module_06_bitnet.bit_linear import BitLinear
from module_06_bitnet.activation_precision import apply_activation_precision
from module_06_bitnet.layer_replacement import (
    replace_linear_layers,
    generate_layer_inspection_report,
)
from module_06_bitnet.model_conversion import convert_fp32_to_bitnet, compute_layer_conversion_stats
from module_06_bitnet.initialization import initialize_bitnet_weights
from module_06_bitnet.checkpointing import save_bitnet_checkpoint, load_bitnet_checkpoint
from module_06_bitnet.qat import BitNetQATTrainer
from module_06_bitnet.pta_baseline import evaluate_ptq_baseline
from module_06_bitnet.evaluation import evaluate_bitnet_model
from module_06_bitnet.profiling import profile_bitnet_model
from module_06_bitnet.comparison import generate_comparison_matrix
from module_06_bitnet.hardware_backend import get_hardware_backend_info, HARDWARE_DISCLAIMER
from module_06_bitnet.experiment import BitNetExperimentRunner, BITNET_ABLATION_CONFIGS
from module_06_bitnet.runtime_specs import HardwareProfile, ARDUINO_UNO_Q_PROFILE, STM32_H7_PROFILE
from module_06_bitnet.export_onnx import export_photon_v0_onnx
from module_06_bitnet.quantize_int8 import quantize_photon_v0_int8
from module_06_bitnet.pack_ternary import pack_model_ternary, quantize_to_ternary, pack_ternary_matrix_to_uint8
from module_06_bitnet.profile_uno_q import profile_for_uno_q, estimate_photon_v0_macs

__all__ = [
    "BitNetConfig",
    "compute_weight_scale",
    "compute_activation_scale",
    "TernarySTEFunction",
    "round_to_ternary",
    "pack_ternary",
    "unpack_ternary",
    "BitLinear",
    "apply_activation_precision",
    "replace_linear_layers",
    "generate_layer_inspection_report",
    "convert_fp32_to_bitnet",
    "compute_layer_conversion_stats",
    "initialize_bitnet_weights",
    "save_bitnet_checkpoint",
    "load_bitnet_checkpoint",
    "BitNetQATTrainer",
    "evaluate_ptq_baseline",
    "evaluate_bitnet_model",
    "profile_bitnet_model",
    "generate_comparison_matrix",
    "get_hardware_backend_info",
    "HARDWARE_DISCLAIMER",
    "BitNetExperimentRunner",
    "BITNET_ABLATION_CONFIGS",
    "HardwareProfile",
    "ARDUINO_UNO_Q_PROFILE",
    "STM32_H7_PROFILE",
    "export_photon_v0_onnx",
    "quantize_photon_v0_int8",
    "pack_model_ternary",
    "quantize_to_ternary",
    "pack_ternary_matrix_to_uint8",
    "profile_for_uno_q",
    "estimate_photon_v0_macs",
]
