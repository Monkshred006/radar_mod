"""Real class-to-class integration test for adaptive diffusion inference."""

from __future__ import annotations

import torch

from module_05_latent_diffusion.latent_diffusion import LatentDiffusionModel
from module_07_adaptive_compute.adaptive_inference import AdaptiveDiffusionInference


class _FixedActionScheduler:
    def __init__(self, action: int):
        self.action = action
        self.calls = 0

    def predict_action(self, state):
        self.calls += 1
        return self.action


def test_adaptive_wrapper_with_real_latent_diffusion_model():
    """Verify the adaptive wrapper reaches the real Module 5 reconstruct path."""
    model = LatentDiffusionModel(
        latent_dim=64,
        hidden_dim=128,
        num_blocks=2,
        timesteps=50,
    )
    model.eval()

    # Use the real model's own latent encoder output as the state source.
    class _RealStateEncoder:
        def __call__(self, z_c, mask):
            state = z_c.mean(dim=1)
            state_dict = {"snr_quality": mask.mean(dim=1)}
            return state, state_dict

    scheduler = _FixedActionScheduler(action=10)
    inference = AdaptiveDiffusionInference(
        diffusion_model=model,
        state_encoder=_RealStateEncoder(),
        scheduler=scheduler,
    )

    x = torch.randn(1, 16, 64)
    output, metadata = inference.reconstruct(x, deterministic=True)

    assert output.shape == (1, 16, 64)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert scheduler.calls == 1
    assert metadata["actions"] == [10]
