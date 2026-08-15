"""Tests proving adaptive inference launches exactly one diffusion trajectory."""

from __future__ import annotations

import torch

from module_07_adaptive_compute.adaptive_inference import AdaptiveDiffusionInference


class _FakeStateEncoder:
    def __call__(self, z_c, mask):
        batch = z_c.shape[0]
        state = torch.zeros(batch, 9)
        state_dict = {"snr_quality": torch.zeros(batch, 1)}
        return state, state_dict


class _FakeScheduler:
    def __init__(self, action: int = 20):
        self.action = action
        self.calls = 0

    def predict_action(self, state):
        self.calls += 1
        return self.action


class _FakeDiffusion:
    def __init__(self):
        self.calls = []

    def encode(self, x):
        return x

    def corruption(self, z_0):
        return z_0, torch.ones_like(z_0)

    def reconstruct(self, x, z_c, mask, num_steps, deterministic):
        self.calls.append(
            {
                "batch_size": x.shape[0],
                "num_steps": num_steps,
                "deterministic": deterministic,
            }
        )
        return z_c, None, None, None


def test_single_sample_runs_exactly_one_diffusion_trajectory():
    diffusion = _FakeDiffusion()
    scheduler = _FakeScheduler(action=20)
    inference = AdaptiveDiffusionInference(
        diffusion_model=diffusion,
        state_encoder=_FakeStateEncoder(),
        scheduler=scheduler,
    )

    x = torch.randn(1, 4)
    output, metadata = inference.reconstruct(x)

    assert output.shape == x.shape
    assert metadata["actions"] == [20]
    assert scheduler.calls == 1
    assert len(diffusion.calls) == 1
    assert diffusion.calls[0]["batch_size"] == 1
    assert diffusion.calls[0]["num_steps"] == 20


def test_batch_samples_each_run_exactly_one_selected_trajectory():
    diffusion = _FakeDiffusion()
    scheduler = _FakeScheduler(action=50)
    inference = AdaptiveDiffusionInference(
        diffusion_model=diffusion,
        state_encoder=_FakeStateEncoder(),
        scheduler=scheduler,
    )

    x = torch.randn(3, 4)
    output, metadata = inference.reconstruct(x)

    assert output.shape == x.shape
    assert metadata["actions"] == [50, 50, 50]
    assert scheduler.calls == 3
    assert len(diffusion.calls) == 3
    assert all(call["batch_size"] == 1 for call in diffusion.calls)
    assert all(call["num_steps"] == 50 for call in diffusion.calls)
