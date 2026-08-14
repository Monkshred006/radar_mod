"""Unit tests for Task Heads and Loss functions."""

import torch
import pytest
from module_04_mamba_hybrid.config import TaskHeadConfig
from module_04_mamba_hybrid.heads import ClassificationHead, RegressionHead, MultiTaskHead
from module_04_mamba_hybrid.losses import get_loss_fn


def test_classification_head():
    cfg = TaskHeadConfig(head_type="classification", num_classes=3)
    head = ClassificationHead(d_model=64, config=cfg)

    x = torch.randn(4, 64)
    logits = head(x)

    assert logits.shape == (4, 3)

    loss_fn = get_loss_fn("cross_entropy")
    targets = torch.tensor([0, 1, 2, 0])
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_regression_head():
    cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=2)
    head = RegressionHead(d_model=64, config=cfg)

    x = torch.randn(4, 64)
    preds = head(x)

    assert preds.shape == (4, 2)

    loss_fn = get_loss_fn("mse")
    targets = torch.randn(4, 2)
    loss = loss_fn(preds, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_multitask_head():
    cfg = TaskHeadConfig(head_type="multitask", num_classes=2, num_regression_outputs=1)
    head = MultiTaskHead(d_model=64, config=cfg)

    x = torch.randn(4, 64)
    out = head(x)

    assert out["logits"].shape == (4, 2)
    assert out["regression"].shape == (4, 1)

    loss_fn = get_loss_fn("multitask")
    targets = {
        "target_class": torch.tensor([0, 1, 0, 1]),
        "target_regression": torch.randn(4, 1),
    }
    loss = loss_fn(out, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
