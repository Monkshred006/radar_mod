"""End-to-end tests for Module 5.

Covers:
1. Full pipeline: Module3 dict → Module4 → Trainer → loss → backward → update
2. Checkpoint save → reload → resume training
3. Overfitting sanity test (loss decreases on tiny dataset)
4. Scene-level split no-leakage verification
5. Evaluator integration after training
"""

import pytest
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader

from module_05_training.config import TrainingConfig
from module_05_training.dataset import (
    make_synthetic_scene_cache,
    PhotonShieldDataset,
    collate_module3,
)
from module_05_training.target_adapter import SyntheticRegressionAdapter
from module_05_training.trainer import Trainer
from module_05_training.evaluator import Evaluator
from module_05_training.checkpointing import save_checkpoint, load_checkpoint


# ── Shared setup ──────────────────────────────────────────────────────────────

WINDOW_LEN = 10
D_MODEL = 32
NUM_LAYERS = 1


def _make_engine_and_head(num_outputs=1):
    from module_04_mamba_hybrid.config import MambaHybridConfig
    from module_04_mamba_hybrid.engine import PhotonMambaHybrid
    from module_04_mamba_hybrid.heads import RegressionHead

    model_cfg = MambaHybridConfig(
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        max_sequence_length=WINDOW_LEN,
    )
    engine = PhotonMambaHybrid(model_cfg)
    from module_04_mamba_hybrid.config import TaskHeadConfig
    head_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=num_outputs)
    head = RegressionHead(D_MODEL, head_cfg)
    return engine, head, model_cfg


def _make_trainer_and_loaders(
    tmp_path,
    num_train_scenes=4,
    num_val_scenes=2,
    frames=50,
    epochs=2,
    lr=1e-3,
):
    engine, head, model_cfg = _make_engine_and_head()
    cfg = TrainingConfig(
        epochs=epochs,
        learning_rate=lr,
        batch_size=4,
        device="cpu",
        num_regression_outputs=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_dir=str(tmp_path / "logs"),
        early_stopping_patience=100,
        scheduler="none",
    )

    adapter = SyntheticRegressionAdapter(num_outputs=1)

    train_cache = make_synthetic_scene_cache(num_scenes=num_train_scenes, frames_per_scene=frames)
    val_cache = make_synthetic_scene_cache(num_scenes=num_val_scenes, frames_per_scene=frames)

    train_ds = PhotonShieldDataset(train_cache, adapter, window_len=WINDOW_LEN, window_stride=5)
    val_ds = PhotonShieldDataset(val_cache, adapter, window_len=WINDOW_LEN, window_stride=5)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_module3)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_module3)

    trainer = Trainer(engine, head, cfg, model_config=model_cfg)
    return trainer, engine, head, cfg, model_cfg, train_loader, val_loader


# ── Test 1: Full pipeline forward → backward → update ─────────────────────────

class TestFullPipeline:
    def test_forward_loss_backward_update(self, tmp_path):
        """Core pipeline: Module3 → Module4 → head → loss → backward → update."""
        trainer, engine, head, cfg, _, train_loader, val_loader = \
            _make_trainer_and_loaders(tmp_path)

        # Snapshot parameters before training
        params_before = {n: p.clone() for n, p in engine.named_parameters()}

        loss, gnorm = trainer.train_epoch(train_loader, epoch=1)

        assert torch.isfinite(torch.tensor(loss)), f"Loss not finite: {loss}"
        assert torch.isfinite(torch.tensor(gnorm)), f"Grad norm not finite: {gnorm}"

        # At least some parameters must have changed
        changed = sum(
            1 for n, p in engine.named_parameters()
            if not torch.allclose(params_before[n], p.detach(), atol=1e-9)
        )
        assert changed > 0, "No engine parameters were updated"

    def test_validation_after_training(self, tmp_path):
        trainer, engine, head, cfg, _, train_loader, val_loader = \
            _make_trainer_and_loaders(tmp_path)
        trainer.train_epoch(train_loader, epoch=1)
        val_loss, metrics = trainer.validate(val_loader)
        assert torch.isfinite(torch.tensor(val_loss))
        assert "mae" in metrics or "accuracy" in metrics  # depends on target_type

    def test_fit_runs_to_completion(self, tmp_path):
        trainer, engine, head, cfg, _, train_loader, val_loader = \
            _make_trainer_and_loaders(tmp_path, epochs=3)
        summary = trainer.fit(train_loader, val_loader)
        assert "best_val_metric" in summary
        assert len(trainer.history) == 3


# ── Test 2: Checkpoint save → reload → resume ─────────────────────────────────

class TestCheckpointResume:
    def test_reload_exact_parameter_match(self, tmp_path):
        """After save/load, model parameters must be bit-exact."""
        trainer, engine, head, cfg, model_cfg, train_loader, val_loader = \
            _make_trainer_and_loaders(tmp_path, epochs=2)
        trainer.fit(train_loader, val_loader)

        ckpt_path = str(tmp_path / "ckpts" / "photonshield_full_mamba_hybrid_latest.pt")
        # If that doesn't exist, save manually
        if not Path(ckpt_path).exists():
            trainer.save_checkpoint(str(tmp_path / "manual.pt"), epoch=2)
            ckpt_path = str(tmp_path / "manual.pt")

        # Rebuild fresh model
        engine2, head2, _ = _make_engine_and_head()
        combined2 = torch.nn.ModuleList([engine2, head2])
        ckpt = load_checkpoint(ckpt_path, combined2)

        # Parameters must match exactly
        combined1 = torch.nn.ModuleList([engine, head])
        for (n1, p1), (n2, p2) in zip(combined1.named_parameters(), combined2.named_parameters()):
            assert torch.allclose(p1.detach(), p2.detach()), f"Mismatch: {n1}"

    def test_resumed_epoch_is_correct(self, tmp_path):
        trainer, engine, head, cfg, model_cfg, train_loader, val_loader = \
            _make_trainer_and_loaders(tmp_path, epochs=2)
        trainer.fit(train_loader, val_loader)

        ckpt_path = str(tmp_path / "resume_test.pt")
        trainer.save_checkpoint(ckpt_path, epoch=2)

        ckpt = load_checkpoint(ckpt_path, torch.nn.ModuleList([engine, head]))
        assert ckpt["epoch"] == 2


# ── Test 3: Overfitting sanity ─────────────────────────────────────────────────

class TestOverfittingSanity:
    def test_loss_decreases_on_tiny_dataset(self, tmp_path):
        """The model should be able to overfit a tiny synthetic dataset.

        This validates: data flow, gradients, loss, optimizer, and target alignment.
        NOT a performance benchmark.
        """
        from module_04_mamba_hybrid.config import MambaHybridConfig
        from module_04_mamba_hybrid.engine import PhotonMambaHybrid
        from module_04_mamba_hybrid.heads import RegressionHead

        # Very small dataset: 2 scenes × 30 frames → ~4 windows
        cache = make_synthetic_scene_cache(
            num_scenes=2, frames_per_scene=30, window_len=WINDOW_LEN
        )
        adapter = SyntheticRegressionAdapter(num_outputs=1)
        ds = PhotonShieldDataset(cache, adapter, window_len=WINDOW_LEN, window_stride=WINDOW_LEN)
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=collate_module3)

        model_cfg = MambaHybridConfig(d_model=D_MODEL, num_layers=NUM_LAYERS, max_sequence_length=WINDOW_LEN)
        engine = PhotonMambaHybrid(model_cfg)
        from module_04_mamba_hybrid.config import TaskHeadConfig
        head_cfg = TaskHeadConfig(head_type="regression", num_regression_outputs=1)
        head = RegressionHead(D_MODEL, head_cfg)

        cfg = TrainingConfig(
            epochs=1,
            learning_rate=1e-2,  # high LR to encourage fast fitting
            batch_size=len(ds),
            device="cpu",
            num_regression_outputs=1,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
            early_stopping_patience=100,
            scheduler="none",
        )
        trainer = Trainer(engine, head, cfg, model_config=model_cfg)

        losses = []
        for epoch in range(15):
            loss, _ = trainer.train_epoch(loader, epoch=epoch + 1)
            losses.append(loss)

        # Loss at end should be lower than at start
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )


# ── Test 4: No data leakage across splits ─────────────────────────────────────

class TestNoDataLeakage:
    def test_scene_splits_are_disjoint(self):
        """Train / val / test scene sets must be fully disjoint."""
        from module_05_training.dataset import make_synthetic_scene_cache

        # Create separate caches with distinct scene IDs
        train_cache = make_synthetic_scene_cache(num_scenes=6, frames_per_scene=40)
        val_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=40)
        test_cache = make_synthetic_scene_cache(num_scenes=2, frames_per_scene=40)

        # The synthetic scenes have IDs like "synthetic_scene_000", "synthetic_scene_001"
        # All three caches share IDs — this represents the SAME scenes going to different splits
        # In real usage, split_dataset() guarantees disjoint scene assignments.
        # Here we verify our test caches themselves are structurally correct.

        train_scene_ids = set(train_cache._cache.keys())
        val_scene_ids = set(val_cache._cache.keys())
        test_scene_ids = set(test_cache._cache.keys())

        # Since make_synthetic_scene_cache uses fixed IDs, same IDs appear.
        # Test the SPLIT API instead (Module 1 split_dataset guarantees disjoint):
        from module_01_radar_input.dataset import RadarDataset, split_dataset
        from module_01_radar_input.config import RadarDatasetConfig
        import tempfile, os

        # Build a synthetic RadarDataset with multiple scenes
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake data files
            import numpy as np
            for scene_i in range(6):
                scene_dir = Path(tmpdir) / f"scene_{scene_i:03d}"
                scene_dir.mkdir(parents=True)
                for frame_i in range(5):
                    np.save(
                        scene_dir / f"frame_{frame_i:04d}.npy",
                        np.zeros((12,), dtype=np.float32)
                    )

            ds_cfg = RadarDatasetConfig(
                dataset_path=tmpdir,
                sequence_length=2,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                random_seed=42,
            )
            ds = RadarDataset(ds_cfg)
            train_ds, val_ds, test_ds = split_dataset(ds)

            # Extract scene IDs from each split's items
            def _get_scene_ids(split_ds):
                return {item.get("scene_id", "default_scene") for item in split_ds.discovered_items}

            train_scenes = _get_scene_ids(train_ds)
            val_scenes = _get_scene_ids(val_ds)
            test_scenes = _get_scene_ids(test_ds)

            assert train_scenes & val_scenes == set(), "Train/Val scene overlap!"
            assert train_scenes & test_scenes == set(), "Train/Test scene overlap!"
            assert val_scenes & test_scenes == set(), "Val/Test scene overlap!"


# ── Test 5: Evaluator after training ─────────────────────────────────────────

class TestEvaluatorIntegration:
    def test_evaluator_post_training(self, tmp_path):
        trainer, engine, head, cfg, _, train_loader, val_loader = \
            _make_trainer_and_loaders(tmp_path, epochs=2)
        trainer.fit(train_loader, val_loader)

        # Use val_loader as stand-in test loader
        ev = Evaluator(cfg)
        results = ev.evaluate(engine, head, val_loader)
        assert results["sample_count"] > 0
        assert torch.isfinite(torch.tensor(results["loss"]))
