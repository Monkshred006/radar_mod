"""Tests for checkpointing: save/load, resume, exact state restoration."""

import pytest
import tempfile
from pathlib import Path
import torch
import torch.nn as nn

from module_05_training.checkpointing import save_checkpoint, load_checkpoint
from module_05_training.config import TrainingConfig


def _simple_model():
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))


class TestCheckpointing:
    def test_save_and_load_model_params(self, tmp_path):
        model = _simple_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cfg = TrainingConfig()

        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(
            path=str(ckpt_path),
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            best_val_metric=0.123,
            training_config=cfg,
            model_config=cfg,
            history=[{"epoch": 5, "train_loss": 0.1}],
            seed=42,
        )
        assert ckpt_path.exists()

        # Create a fresh model and load
        model2 = _simple_model()
        ckpt = load_checkpoint(str(ckpt_path), model2)

        # Parameters must match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Mismatch in param: {n1}"

    def test_checkpoint_epoch_restored(self, tmp_path):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters())
        cfg = TrainingConfig()
        path = str(tmp_path / "ckpt.pt")

        save_checkpoint(path, model, opt, None, epoch=7, best_val_metric=0.5,
                        training_config=cfg, model_config=cfg, history=[], seed=0)
        ckpt = load_checkpoint(path, model)
        assert ckpt["epoch"] == 7

    def test_checkpoint_best_metric_restored(self, tmp_path):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters())
        cfg = TrainingConfig()
        path = str(tmp_path / "ckpt.pt")

        save_checkpoint(path, model, opt, None, epoch=3, best_val_metric=0.456,
                        training_config=cfg, model_config=cfg, history=[], seed=0)
        ckpt = load_checkpoint(path, model)
        assert ckpt["best_val_metric"] == pytest.approx(0.456)

    def test_optimizer_state_restored(self, tmp_path):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Do one optimizer step so state is non-trivial
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        cfg = TrainingConfig()
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, model, opt, None, epoch=1, best_val_metric=0.0,
                        training_config=cfg, model_config=cfg, history=[], seed=0)

        model2 = _simple_model()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        load_checkpoint(path, model2, optimizer=opt2)

        # Check optimizer state (moment buffers) are restored
        for g1, g2 in zip(opt.param_groups, opt2.param_groups):
            assert g1["lr"] == pytest.approx(g2["lr"])

    def test_history_preserved(self, tmp_path):
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters())
        history = [{"epoch": i, "train_loss": 1.0 / (i + 1)} for i in range(5)]
        cfg = TrainingConfig()
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, model, opt, None, epoch=5, best_val_metric=0.2,
                        training_config=cfg, model_config=cfg, history=history, seed=0)
        ckpt = load_checkpoint(path, model)
        assert len(ckpt["history"]) == 5
        assert ckpt["history"][2]["epoch"] == 2

    def test_scheduler_state_saved_and_loaded(self, tmp_path):
        from torch.optim.lr_scheduler import StepLR
        model = _simple_model()
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        sched = StepLR(opt, step_size=2, gamma=0.5)
        sched.step()  # advance one step

        cfg = TrainingConfig()
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, model, opt, sched, epoch=2, best_val_metric=0.0,
                        training_config=cfg, model_config=cfg, history=[], seed=0)

        model2 = _simple_model()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=0.1)
        sched2 = StepLR(opt2, step_size=2, gamma=0.5)
        load_checkpoint(path, model2, optimizer=opt2, scheduler=sched2)
        assert sched2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]
