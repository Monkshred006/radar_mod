"""Unit tests for Model Checkpointing."""

import tempfile
import os
import torch
import pytest
from module_04_mamba_hybrid.config import MambaHybridConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_04_mamba_hybrid.checkpointing import save_checkpoint, load_checkpoint


def test_save_and_load_checkpoint():
    config = MambaHybridConfig(d_model=32, num_layers=1, backend="fallback")
    engine = PhotonMambaHybrid(config)
    optimizer = torch.optim.Adam(engine.parameters(), lr=1e-3)

    tmp_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmp_dir, "model_ckpt.pt")

    # Save
    save_checkpoint(
        filepath=ckpt_path,
        model=engine,
        config=config,
        optimizer=optimizer,
        epoch=5,
        metrics={"val_loss": 0.25},
    )

    assert os.path.exists(ckpt_path)

    # Load into fresh instance
    new_engine = PhotonMambaHybrid(config)
    new_optimizer = torch.optim.Adam(new_engine.parameters(), lr=1e-3)

    loaded_info = load_checkpoint(ckpt_path, new_engine, optimizer=new_optimizer)

    assert loaded_info["epoch"] == 5
    assert loaded_info["metrics"]["val_loss"] == 0.25

    # Verify parameters match exactly
    for p1, p2 in zip(engine.parameters(), new_engine.parameters()):
        assert torch.allclose(p1, p2)
