"""BitLinear Modular Low-Bit Linear Layer for Module 6.

A drop-in replacement for nn.Linear using Straight-Through Estimator (STE)
ternary weight quantization and per-tensor weight scaling.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.ternary import round_to_ternary
from module_06_bitnet.activation_precision import apply_activation_precision


class BitLinear(nn.Module):
    """BitNet-Style Low-Bit Linear Layer.

    Replaces standard nn.Linear. Maintains trainable FP32 master weights during training.
    In the forward pass, ternarizes weights W ∈ {-α, 0, +α} using STE autograd.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to False, layer will not learn an additive bias. Default: True.
        config: BitNetConfig specifying scaling method, activation precision, etc.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[BitNetConfig] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or BitNetConfig()

        # FP32 master weight parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize master weights using Kaiming Uniform."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [..., in_features].

        Returns:
            Output tensor [..., out_features].
        """
        if not self.config.enabled:
            return F.linear(x, self.weight, self.bias)

        # 1. Apply activation precision
        x_prec = apply_activation_precision(x, self.config.activation_precision)

        # 2. Quantize master weight to ternary representation via STE
        w_quant, scale, _ = round_to_ternary(
            self.weight,
            scale_method=self.config.scaling_method,
            scale_scope=self.config.scaling_scope,
        )

        # 3. Perform linear computation
        return F.linear(x_prec, w_quant, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear_module: nn.Linear,
        config: Optional[BitNetConfig] = None,
    ) -> BitLinear:
        """Factory: Convert an existing nn.Linear instance into a BitLinear layer.

        Copies parameter data from the original linear_module into master weights.

        Args:
            linear_module: Source nn.Linear instance.
            config: BitNetConfig to configure the new BitLinear layer.

        Returns:
            A new BitLinear instance initialized with the linear_module's weights.
        """
        bit_layer = cls(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            bias=linear_module.bias is not None,
            config=config,
        )
        with torch.no_grad():
            bit_layer.weight.copy_(linear_module.weight.data)
            if linear_module.bias is not None and bit_layer.bias is not None:
                bit_layer.bias.copy_(linear_module.bias.data)

        return bit_layer

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, scaling_method='{self.config.scaling_method}', "
            f"scaling_scope='{self.config.scaling_scope}', "
            f"activation_precision='{self.config.activation_precision}'"
        )
