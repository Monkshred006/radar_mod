"""Tests for LR scheduler — standalone file (scheduler tests also in test_optimizer.py)."""

import pytest
from module_05_training.config import TrainingConfig
from module_05_training.optimizer import get_optimizer
from module_05_training.scheduler import get_scheduler, NoScheduler
import torch.nn as nn


def _opt(lr=1e-3):
    model = nn.Linear(4, 2)
    cfg = TrainingConfig(learning_rate=lr)
    return get_optimizer(model, cfg)


class TestSchedulerStepping:
    """Additional scheduler stepping/state tests."""

    def test_step_lr_halves(self):
        opt = _opt(lr=0.1)
        cfg = TrainingConfig(scheduler="step", scheduler_step_size=1, scheduler_gamma=0.5)
        sched = get_scheduler(opt, cfg)
        sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.05, rel=1e-4)

    def test_none_keeps_lr_constant(self):
        opt = _opt(lr=0.01)
        cfg = TrainingConfig(scheduler="none")
        sched = get_scheduler(opt, cfg)
        for _ in range(5):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01, rel=1e-6)

    def test_plateau_state_dict_round_trip(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        opt = _opt()
        cfg = TrainingConfig(scheduler="plateau")
        sched = get_scheduler(opt, cfg)
        sd = sched.state_dict()
        assert isinstance(sd, dict)
        sched.load_state_dict(sd)  # should not raise
