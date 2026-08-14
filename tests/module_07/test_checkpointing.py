"""Tests for Module 7 decision head checkpointing."""

import pytest
import torch
from pathlib import Path

from module_07_decision.config import DecisionModelConfig, DecisionConfig
from module_07_decision.multitask import PhotonShieldMultiTask
from module_07_decision.checkpointing import save_decision_checkpoint, load_decision_checkpoint


class TestDecisionCheckpointing:
    def test_save_load_roundtrip(self, tmp_path):
        m_cfg = DecisionModelConfig(d_model=32)
        d_cfg = DecisionConfig(target_threshold=0.6, anomaly_threshold=0.4)
        model1 = PhotonShieldMultiTask(m_cfg)

        ckpt_path = str(tmp_path / "module07_decision.pt")

        save_decision_checkpoint(
            path=ckpt_path,
            multi_task_model=model1,
            model_config=m_cfg,
            decision_config=d_cfg,
            epoch=2,
            metrics={"f1": 0.95},
        )
        assert Path(ckpt_path).exists()

        model2 = PhotonShieldMultiTask(m_cfg)
        payload = load_decision_checkpoint(ckpt_path, model2)

        assert payload["epoch"] == 2
        assert payload["decision_config"]["target_threshold"] == 0.6

        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2)
