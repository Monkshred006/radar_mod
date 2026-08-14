"""RL Decision State Builder for Sensor Control (Phase V2 Preparation).

Constructs a compact, causal state vector from PhotonV0 perception outputs
(detection confidence, class probabilities, anomaly score, pooled latent) and
current sensor configuration.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionStateBuilder:
    """Builds compact state vectors for RL sensor control policy."""

    def __init__(
        self,
        include_latent: bool = True,
        latent_dim: int = 64,
        latent_summary_dim: int = 8,
        num_classes: int = 4,
        include_telemetry: bool = True,
    ) -> None:
        self.include_latent = include_latent
        self.latent_dim = latent_dim
        self.latent_summary_dim = latent_summary_dim
        self.num_classes = num_classes
        self.include_telemetry = include_telemetry

        # Linear compression for latent vector if included
        if include_latent and latent_dim > latent_summary_dim:
            self.latent_compress = nn.Linear(latent_dim, latent_summary_dim)
        else:
            self.latent_compress = nn.Identity()

    @property
    def state_dim(self) -> int:
        """Calculate total RL state dimension."""
        # 1 (detection) + num_classes (class probs) + 1 (anomaly)
        dim = 1 + self.num_classes + 1
        if self.include_latent:
            dim += self.latent_summary_dim
        if self.include_telemetry:
            # gain, pulse_width, sampling_rate, frame_avg, estimated_snr
            dim += 5
        return dim

    def build_state(
        self,
        perception_outputs: Dict[str, torch.Tensor],
        sensor_telemetry: Optional[Dict[str, float]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Construct state tensor [B, state_dim].

        Args:
            perception_outputs: Dictionary with keys:
                - 'detection': [B, 1]
                - 'classification': [B, num_classes] (logits or probs)
                - 'anomaly': [B, 1]
                - (optional) 'pooled_latent': [B, H]
            sensor_telemetry: Dict with current sensor parameters.
            device: Target torch device.

        Returns:
            State tensor `[B, state_dim]`.
        """
        # Ensure detached perception outputs (no gradient propagation back to Mamba)
        det = perception_outputs["detection"].detach().to(device)  # [B, 1]
        cls_logits = perception_outputs["classification"].detach().to(device)
        cls_probs = F.softmax(cls_logits, dim=-1)  # [B, C]
        ano = perception_outputs["anomaly"].detach().to(device)  # [B, 1]

        components = [det, cls_probs, ano]

        if self.include_latent:
            pooled = perception_outputs.get(
                "pooled_latent", torch.zeros(det.shape[0], self.latent_dim, device=device)
            ).detach().to(device)
            # Compress to compact summary
            with torch.no_grad():
                latent_sum = self.latent_compress(pooled)
            components.append(latent_sum)

        if self.include_telemetry:
            B = det.shape[0]
            if sensor_telemetry is None:
                sensor_telemetry = {
                    "gain_db": 0.0,
                    "pulse_width_us": 10.0,
                    "sampling_rate_mhz": 20.0,
                    "frame_averaging": 1.0,
                    "snr_db": 10.0,
                }

            telem_vec = torch.tensor(
                [
                    sensor_telemetry.get("gain_db", 0.0) / 30.0,  # Normalized [-1, 1]
                    sensor_telemetry.get("pulse_width_us", 10.0) / 50.0,
                    sensor_telemetry.get("sampling_rate_mhz", 20.0) / 100.0,
                    sensor_telemetry.get("frame_averaging", 1.0) / 8.0,
                    sensor_telemetry.get("snr_db", 10.0) / 40.0,
                ],
                dtype=torch.float32,
                device=device,
            ).repeat(B, 1)  # [B, 5]
            components.append(telem_vec)

        return torch.cat(components, dim=-1)
