"""Unit tests for PhotonV0 minimal Mamba temporal perception architecture."""

import pytest
import torch
import torch.nn as nn

from module_04_mamba_hybrid.photon_v0 import (
    PhotonV0,
    DetectionHead,
    ClassificationHead,
    AnomalyHead,
    count_parameters,
)
from module_04_mamba_hybrid.mamba_core import PurePyTorchSSM, MiniMambaBlock
from module_04_mamba_hybrid.mamba_attention import MambaAttentionHybridBlock


class TestPhotonV0Architecture:
    """Test suite for PhotonV0 and its components."""

    def test_pure_pytorch_ssm_shapes(self):
        B, T, D = 4, 16, 64
        ssm = PurePyTorchSSM(d_model=D, d_state=16, d_conv=4, expand=2)
        x = torch.randn(B, T, D)
        out = ssm(x)
        assert out.shape == (B, T, D)
        assert not torch.isnan(out).any()

    def test_mini_mamba_block_forward(self):
        B, T, D = 2, 16, 64
        block = MiniMambaBlock(d_model=D, d_state=16, backend="fallback")
        x = torch.randn(B, T, D)
        out = block(x)
        assert out.shape == (B, T, D)
        assert not torch.isnan(out).any()

    def test_mamba_attention_hybrid_block(self):
        B, T, D = 2, 16, 64
        # Without attention (V0 default)
        block_v0 = MambaAttentionHybridBlock(d_model=D, use_attention=False, backend="fallback")
        out_v0 = block_v0(torch.randn(B, T, D))
        assert out_v0.shape == (B, T, D)

        # With attention (V1/V2 future hook)
        block_v1 = MambaAttentionHybridBlock(d_model=D, use_attention=True, backend="fallback")
        out_v1 = block_v1(torch.randn(B, T, D))
        assert out_v1.shape == (B, T, D)

    def test_heads_forward(self):
        B, H = 8, 64
        pooled = torch.randn(B, H)

        det_head = DetectionHead(hidden_dim=H)
        cls_head = ClassificationHead(hidden_dim=H, num_classes=4)
        ano_head = AnomalyHead(hidden_dim=H)

        det_out = det_head(pooled)
        cls_out = cls_head(pooled)
        ano_out = ano_head(pooled)

        assert det_out.shape == (B, 1)
        assert (det_out >= 0.0).all() and (det_out <= 1.0).all()
        assert cls_out.shape == (B, 4)
        assert ano_out.shape == (B, 1)

    def test_photon_v0_forward_and_latents(self):
        B, T, D = 4, 16, 64
        model = PhotonV0(
            input_dim=D,
            hidden_dim=64,
            num_layers=2,
            sequence_length=T,
            num_classes=4,
            backend="fallback",
        )
        x = torch.randn(B, T, D)

        outputs = model(x, return_latents=True)
        assert "detection" in outputs
        assert "classification" in outputs
        assert "anomaly" in outputs
        assert "latent" in outputs
        assert "pooled_latent" in outputs

        assert outputs["detection"].shape == (B, 1)
        assert outputs["classification"].shape == (B, 4)
        assert outputs["anomaly"].shape == (B, 1)
        assert outputs["latent"].shape == (B, T, 64)
        assert outputs["pooled_latent"].shape == (B, 64)

    def test_parameter_count_utility(self):
        model = PhotonV0(input_dim=64, hidden_dim=64, num_layers=2, backend="fallback")
        n_params = count_parameters(model)
        assert isinstance(n_params, int)
        assert n_params > 10_000
        assert model.count_parameters() == n_params
