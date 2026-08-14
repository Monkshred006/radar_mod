"""Tests for Trainer: forward, backward, parameter update, gradient management."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from module_05_training.config import TrainingConfig
from module_05_training.losses import TrainingNaNError
from module_05_training.dataset import make_synthetic_scene_cache, PhotonShieldDataset, collate_module3
from module_05_training.target_adapter import SyntheticRegressionAdapter
from module_05_training.trainer import Trainer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_loader(num_scenes=3, frames=40, window_len=10, batch_size=4):
    """Build a DataLoader from synthetic SceneFeatureCache."""
    cache = make_synthetic_scene_cache(
        num_scenes=num_scenes, frames_per_scene=frames, window_len=window_len
    )
    adapter = SyntheticRegressionAdapter(num_outputs=1)
    ds = PhotonShieldDataset(cache, adapter, window_len=window_len, window_stride=5)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_module3)


def _make_trainer(epochs=2, lr=1e-3, gradient_clip=1.0, num_outputs=1,
                  window_len=10, d_model=32, num_layers=1):
    from module_04_mamba_hybrid.config import MambaHybridConfig
    from module_04_mamba_hybrid.engine import PhotonMambaHybrid
    from module_04_mamba_hybrid.heads import RegressionHead

    cfg = TrainingConfig(
        epochs=epochs,
        learning_rate=lr,
        gradient_clip_norm=gradient_clip,
        batch_size=4,
        num_regression_outputs=num_outputs,
        checkpoint_dir="test_checkpoints",
        log_dir="test_logs",
        val_every_n_epochs=1,
        early_stopping_patience=100,  # effectively disabled
        scheduler="none",
        device="cpu",
    )
    model_cfg = MambaHybridConfig(
        d_model=d_model,
        num_layers=num_layers,
        max_sequence_length=window_len,
    )
    engine = PhotonMambaHybrid(model_cfg)
    from module_04_mamba_hybrid.config import TaskHeadConfig
    head_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=num_outputs)
    head = RegressionHead(d_model, head_cfg)
    trainer = Trainer(engine, head, cfg, model_config=model_cfg)
    return trainer, engine, head, cfg, model_cfg


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTrainerForwardBackward:
    def test_forward_backward_cpu(self):
        """Forward pass works, loss is finite, params receive gradients."""
        trainer, engine, head, cfg, _ = _make_trainer()
        loader = _make_loader()
        loss, gnorm = trainer.train_epoch(loader, epoch=1)
        assert torch.isfinite(torch.tensor(loss))
        assert torch.isfinite(torch.tensor(gnorm))

    def test_params_update_after_step(self):
        """Parameters change after one training step."""
        trainer, engine, head, cfg, _ = _make_trainer(epochs=1)
        loader = _make_loader()

        params_before = {
            name: p.clone().detach()
            for name, p in list(engine.named_parameters()) + list(head.named_parameters())
            if p.requires_grad
        }
        trainer.train_epoch(loader, epoch=1)

        changed = 0
        for name, p_before in params_before.items():
            # Get current param
            all_params = dict(list(engine.named_parameters()) + list(head.named_parameters()))
            p_after = all_params[name]
            if not torch.allclose(p_before, p_after.detach(), atol=1e-8):
                changed += 1
        assert changed > 0, "No parameters were updated"

    def test_validation_runs(self):
        trainer, engine, head, cfg, _ = _make_trainer()
        loader = _make_loader()
        val_loss, metrics = trainer.validate(loader)
        assert torch.isfinite(torch.tensor(val_loss))
        assert isinstance(metrics, dict)


class TestGradientManagement:
    def test_gradient_clipping_applied(self):
        """Gradient norm returns original norm, and clipped gradients are <= clip value."""
        trainer, engine, head, cfg, _ = _make_trainer(
            lr=1e-3, gradient_clip=0.01
        )
        loader = _make_loader()
        loss, gnorm = trainer.train_epoch(loader, epoch=1)
        assert gnorm > 0  # returns unclipped L2 norm
        # Verify actual post-clipping norm of parameters is clipped
        post_clip_norm = sum(
            p.grad.norm().item() ** 2 for p in trainer._combined.parameters()
            if p.grad is not None
        ) ** 0.5
        assert post_clip_norm <= 0.01 + 1e-5

    def test_nan_gradient_raises(self):
        """NaN in gradients raises TrainingNaNError."""
        trainer, engine, head, cfg, _ = _make_trainer()
        loader = _make_loader()

        # Corrupt a parameter gradient artificially
        def _corrupt_hook(grad):
            return torch.full_like(grad, float("nan"))

        hooks = []
        for p in engine.parameters():
            if p.requires_grad:
                hooks.append(p.register_hook(_corrupt_hook))
                break

        try:
            with pytest.raises(TrainingNaNError):
                trainer.train_epoch(loader, epoch=1)
        finally:
            for h in hooks:
                h.remove()


class TestFitLoop:
    def test_fit_completes(self):
        trainer, engine, head, cfg, _ = _make_trainer(epochs=2)
        train_loader = _make_loader()
        val_loader = _make_loader()
        summary = trainer.fit(train_loader, val_loader)
        assert "best_val_metric" in summary
        assert len(trainer.history) == 2
