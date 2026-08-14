"""Tests for optimizer and scheduler factories."""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD

from module_05_training.config import TrainingConfig
from module_05_training.optimizer import get_optimizer
from module_05_training.scheduler import get_scheduler, NoScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


def _simple_model():
    return nn.Linear(4, 2)


class TestOptimizer:
    def test_adamw_default(self):
        model = _simple_model()
        cfg = TrainingConfig(optimizer="adamw", learning_rate=1e-3, weight_decay=1e-2)
        opt = get_optimizer(model, cfg)
        assert isinstance(opt, AdamW)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_adam(self):
        model = _simple_model()
        cfg = TrainingConfig(optimizer="adam", learning_rate=5e-4)
        opt = get_optimizer(model, cfg)
        assert isinstance(opt, Adam)

    def test_sgd(self):
        model = _simple_model()
        cfg = TrainingConfig(optimizer="sgd", learning_rate=0.01, sgd_momentum=0.9)
        opt = get_optimizer(model, cfg)
        assert isinstance(opt, SGD)
        assert opt.param_groups[0]["momentum"] == pytest.approx(0.9)

    def test_unknown_raises(self):
        model = _simple_model()
        cfg = TrainingConfig(optimizer="newtonian_descent")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer(model, cfg)

    def test_only_requires_grad_params(self):
        model = nn.Linear(4, 2)
        model.bias.requires_grad = False
        cfg = TrainingConfig()
        opt = get_optimizer(model, cfg)
        total_params = sum(len(g["params"]) for g in opt.param_groups)
        # Only weight (not bias) should be in optimizer
        assert total_params == 1


class TestScheduler:
    def _opt(self, lr=1e-3):
        model = _simple_model()
        cfg = TrainingConfig(learning_rate=lr)
        return get_optimizer(model, cfg)

    def test_cosine(self):
        opt = self._opt()
        cfg = TrainingConfig(scheduler="cosine", epochs=10)
        sched = get_scheduler(opt, cfg)
        assert isinstance(sched, CosineAnnealingLR)

    def test_step(self):
        opt = self._opt()
        cfg = TrainingConfig(scheduler="step", scheduler_step_size=5, scheduler_gamma=0.5)
        sched = get_scheduler(opt, cfg)
        assert isinstance(sched, StepLR)

    def test_plateau(self):
        opt = self._opt()
        cfg = TrainingConfig(scheduler="plateau")
        sched = get_scheduler(opt, cfg)
        assert isinstance(sched, ReduceLROnPlateau)

    def test_none(self):
        opt = self._opt()
        cfg = TrainingConfig(scheduler="none")
        sched = get_scheduler(opt, cfg)
        assert isinstance(sched, NoScheduler)

    def test_cosine_lr_decreases(self):
        opt = self._opt(lr=0.1)
        cfg = TrainingConfig(scheduler="cosine", epochs=5, scheduler_min_lr=0.0)
        sched = get_scheduler(opt, cfg)
        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(3):
            sched.step()
        assert opt.param_groups[0]["lr"] < initial_lr

    def test_unknown_raises(self):
        opt = self._opt()
        cfg = TrainingConfig(scheduler="cyclical_chaos")
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler(opt, cfg)
