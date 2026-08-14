"""Temperature Scaling Probability Calibration for Module 7 (Validation Set Only)."""

from __future__ import annotations
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Temperature Scaler for probability calibration.

    Calibrates logits by dividing by scalar temperature T: z_calibrated = z / T.
    Fits T strictly on VALIDATION dataset to prevent data leakage.
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([initial_temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Unnormalized logits tensor [N, C] or [N, 1].

        Returns:
            Calibrated logits tensor [N, C] or [N, 1].
        """
        # Clamp temperature to avoid zero division or negative scaling
        t = torch.clamp(self.temperature, min=1e-3)
        return logits / t

    def fit_validation(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        is_binary: bool = False,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Fit optimal temperature scalar T on validation dataset.

        Args:
            val_logits: Validation unnormalized logits tensor [N, C] or [N, 1].
            val_labels: Validation ground truth targets [N].
            is_binary: If True, uses BCEWithLogitsLoss; else CrossEntropyLoss.
            lr: Learning rate for L-BFGS or Adam optimizer.
            max_iter: Optimization iterations.

        Returns:
            Fitted temperature parameter value (float).
        """
        self.to(val_logits.device)
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        val_labels_float = val_labels.float()

        def closure():
            optimizer.zero_grad()
            calibrated = self.forward(val_logits)
            if is_binary:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    calibrated.view(-1), val_labels_float.view(-1)
                )
            else:
                loss = nn.functional.cross_entropy(calibrated, val_labels.long())
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.get_temperature())

    def get_temperature(self) -> float:
        """Return scalar temperature value."""
        return float(torch.clamp(self.temperature, min=1e-3).item())
