"""Quantization-Aware Training (QAT) for Module 6 BitNet models.

Reuses Module 5's Trainer engine for clean, non-duplicated QAT fine-tuning.
STE autograd propagates loss gradients directly to FP32 master weights.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module_05_training.config import TrainingConfig
from module_05_training.trainer import Trainer
from module_06_bitnet.config import BitNetConfig
from module_06_bitnet.checkpointing import save_bitnet_checkpoint


class BitNetQATTrainer:
    """Trainer wrapper for Quantization-Aware Training (QAT).

    Args:
        engine: BitNet Module 4 engine containing BitLinear layers.
        head: BitNet task head.
        bitnet_config: BitNetConfig.
        train_config: TrainingConfig (reused from Module 5).
        model_config: MambaHybridConfig.
        source_fp32_checkpoint: Path to reference FP32 checkpoint.
    """

    def __init__(
        self,
        engine: nn.Module,
        head: nn.Module,
        bitnet_config: BitNetConfig,
        train_config: TrainingConfig,
        model_config: Any,
        source_fp32_checkpoint: str = "",
    ):
        self.engine = engine
        self.head = head
        self.bitnet_config = bitnet_config
        self.train_config = train_config
        self.model_config = model_config
        self.source_fp32_checkpoint = source_fp32_checkpoint

        # Reuse Module 5's Trainer engine
        self.trainer = Trainer(
            engine=engine,
            head=head,
            config=train_config,
            model_config=model_config,
            experiment_name="bitnet_qat",
        )

    def train_qat(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Run QAT fine-tuning loop.

        Args:
            train_loader: Training DataLoader from Module 5.
            val_loader: Validation DataLoader from Module 5.

        Returns:
            Summary report dictionary.
        """
        print("[BitNet QAT] Starting Quantization-Aware Training fine-tuning...")
        summary = self.trainer.fit(train_loader, val_loader)

        # Save final BitNet QAT checkpoint
        save_path = f"{self.bitnet_config.checkpoint_dir}/bitnet_qat_final.pt"
        save_bitnet_checkpoint(
            path=save_path,
            engine=self.engine,
            head=self.head,
            bitnet_config=self.bitnet_config,
            model_config=self.model_config,
            training_config=self.train_config,
            source_fp32_checkpoint=self.source_fp32_checkpoint,
            optimizer=self.trainer.optimizer,
            epoch=self.train_config.epochs,
            metrics={"best_val_metric": summary.get("best_val_metric", float("inf"))},
        )
        summary["checkpoint_path"] = save_path
        print(f"[BitNet QAT] Fine-tuning complete. Checkpoint saved: {save_path}")
        return summary
