"""Top-level Latent Diffusion Model for PhotonShield AI V1.

Binds the frozen PhotonV0 encoder, conditional temporal denoiser, DDPM scheduler,
and radar corruption operators.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import yaml

from module_04_mamba_hybrid.photon_v0 import PhotonV0
from module_05_latent_diffusion.denoiser import LightweightDenoiser
from module_05_latent_diffusion.scheduler import DDPMScheduler
from module_05_latent_diffusion.corruption import RadarLatentCorruption
from module_05_latent_diffusion.losses import DiffusionLoss


class LatentDiffusionModel(nn.Module):
    """Conditional Latent Diffusion Model operating on representations from frozen PhotonV0."""

    def __init__(
        self,
        v0_checkpoint_path: Optional[Union[str, Path]] = None,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        timesteps: int = 50,
        corruption_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps

        # 1. Instantiate Frozen PhotonV0 Encoder
        self.encoder = PhotonV0(
            input_dim=latent_dim,
            hidden_dim=latent_dim,
            num_layers=2,
            sequence_length=16,
            num_classes=4,
            use_attention=False,
        )

        if v0_checkpoint_path:
            ckpt_path = Path(v0_checkpoint_path)
            if ckpt_path.exists():
                state_dict = torch.load(ckpt_path, map_location="cpu")
                self.encoder.load_state_dict(state_dict)
                print(f"[LatentDiffusionModel] Successfully loaded frozen PhotonV0 checkpoint from '{ckpt_path}'")
            else:
                print(f"[LatentDiffusionModel] WARNING: Checkpoint path '{ckpt_path}' not found, using initialized weights.")

        # Freeze encoder completely
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2. Trainable Lightweight Denoiser
        self.denoiser = LightweightDenoiser(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )

        # 3. Diffusion Noise Scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=timesteps)

        # 4. Latent State Corruption Pipeline
        self.corruption = RadarLatentCorruption(corruption_config)

        # 5. Loss Module
        self.loss_fn = DiffusionLoss()

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract clean latent sequence representation z_0 [B, T, D] from frozen PhotonV0 encoder."""
        self.encoder.eval()
        latent_seq, _ = self.encoder.extract_latents(x)
        return latent_seq

    def forward(
        self,
        x: torch.Tensor,
        z_c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute diffusion training step loss.

        Args:
            x: Raw radar feature batch [B, T, 64].
            z_c: Optional pre-corrupted latent tensor [B, T, 64]. If None, corrupted via self.corruption.

        Returns:
            Tuple of (loss, z_0, z_c, eps_pred)
        """
        # 1. Extract clean latent z_0 using frozen encoder
        with torch.no_grad():
            z_0 = self.encode(x)
            if z_c is None:
                z_c = self.corruption(z_0)

        # 2. Sample random timesteps uniformly
        B = z_0.shape[0]
        device = z_0.device
        t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)

        # 3. Sample random Gaussian noise epsilon
        epsilon = torch.randn_like(z_0)

        # 4. Add noise to clean latent according to schedule
        z_t = self.scheduler.add_noise(original_samples=z_0, noise=epsilon, timesteps=t)

        # 5. Predict noise using conditional denoiser
        eps_pred = self.denoiser(z_t=z_t, condition=z_c, timestep=t)

        # 6. Compute MSE noise loss
        loss = self.loss_fn(eps_pred, epsilon)

        return loss, z_0, z_c, eps_pred

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        z_c: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct clean latent state z_hat from input or corrupted state.

        Args:
            x: Input radar tensor [B, T, 64].
            z_c: Optional pre-corrupted latent tensor.
            num_steps: Denoising steps.

        Returns:
            Tuple of (z_hat, z_0, z_c)
        """
        z_0 = self.encode(x)
        if z_c is None:
            z_c = self.corruption(z_0)

        z_hat = self.scheduler.reconstruct(
            denoiser=self.denoiser,
            condition=z_c,
            num_inference_steps=num_steps or self.timesteps,
        )

        return z_hat, z_0, z_c
