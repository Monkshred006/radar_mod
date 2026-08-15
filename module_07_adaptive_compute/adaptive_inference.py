"""Single-pass adaptive diffusion inference for PhotonShield V3.

Unlike the V3 evaluation/oracle experiment, this module chooses the diffusion
step budget BEFORE running diffusion and then launches exactly one diffusion
trajectory per input sample.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from module_05_latent_diffusion.latent_diffusion import LatentDiffusionModel
from module_07_adaptive_compute.action_space import ACTIONS
from module_07_adaptive_compute.state_encoder import AdaptiveComputeStateEncoder


class AdaptiveDiffusionInference:
    """Connect a V3 scheduler to the real latent-diffusion inference path.

    The scheduler must expose ``predict_action(state)`` and return one of the
    supported diffusion budgets in ``ACTIONS``.
    """

    def __init__(
        self,
        diffusion_model: LatentDiffusionModel,
        state_encoder: AdaptiveComputeStateEncoder,
        scheduler: Any,
    ) -> None:
        self.diffusion_model = diffusion_model
        self.state_encoder = state_encoder
        self.scheduler = scheduler

    @staticmethod
    def _validate_action(action: int) -> int:
        action = int(action)
        if action not in ACTIONS:
            raise ValueError(f"Scheduler returned unsupported diffusion budget: {action}. Expected one of {ACTIONS}.")
        return action

    @torch.no_grad()
    def predict_action(
        self,
        z_c: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[List[int], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """Choose diffusion budgets from corrupted inputs without running diffusion."""
        state, state_dict = self.state_encoder(z_c, mask)
        actions: List[int] = []
        for i in range(state.shape[0]):
            action = self.scheduler.predict_action(state[i])
            if isinstance(action, tuple):
                action = action[0]
            actions.append(self._validate_action(action))

        per_sample_state = [
            {name: values[i : i + 1] for name, values in state_dict.items()}
            for i in range(state.shape[0])
        ]
        return actions, state, per_sample_state

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        z_c: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run single-pass adaptive reconstruction.

        If ``z_c`` and ``mask`` are not supplied, they are generated once from
        the clean latent representation. The scheduler then sees only this
        corrupted/observed information and selects the diffusion budget before
        any reverse diffusion trajectory is launched.

        For a batch containing different actions, samples are reconstructed
        independently so each sample still executes exactly one trajectory.
        """
        if z_c is None or mask is None:
            z_0 = self.diffusion_model.encode(x)
            z_c, mask = self.diffusion_model.corruption(z_0)
        else:
            z_0 = None

        actions, state, state_dicts = self.predict_action(z_c, mask)

        outputs: List[torch.Tensor] = []
        for i, action in enumerate(actions):
            z_hat, _, _, _ = self.diffusion_model.reconstruct(
                x[i : i + 1],
                z_c=z_c[i : i + 1],
                mask=mask[i : i + 1],
                num_steps=action,
                deterministic=deterministic,
            )
            outputs.append(z_hat)

        z_hat = torch.cat(outputs, dim=0)
        metadata: Dict[str, Any] = {
            "actions": actions,
            "state": state,
            "state_dict": state_dicts,
            "z_c": z_c,
            "mask": mask,
            "z_0": z_0,
        }
        return z_hat, metadata
