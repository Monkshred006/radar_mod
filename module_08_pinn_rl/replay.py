"""Experience replay and rollout buffers for Module 8.

ReplayBuffer  — off-policy experience replay (uniform random sampling).
RolloutBuffer — on-policy rollout for PPO including GAE advantage estimation.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

from module_08_pinn_rl.transitions import Transition


class ReplayBuffer:
    """Circular replay buffer for off-policy algorithms.

    Stores (state, action, reward, next_state, done) tuples as tensors.

    Parameters
    ----------
    state_dim : int
    action_dim : int
        Encoded action dimension (one-hot for discrete).
    max_size : int
    device : str
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 10_000,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros(max_size, state_dim)
        self.actions = torch.zeros(max_size, action_dim)
        self.rewards = torch.zeros(max_size, 1)
        self.next_states = torch.zeros(max_size, state_dim)
        self.dones = torch.zeros(max_size, 1)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = torch.from_numpy(state.astype(np.float32))
        self.actions[self.ptr] = torch.from_numpy(action.astype(np.float32))
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = torch.from_numpy(next_state.astype(np.float32))
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": self.states[idx].to(self.device),
            "actions": self.actions[idx].to(self.device),
            "rewards": self.rewards[idx].to(self.device),
            "next_states": self.next_states[idx].to(self.device),
            "dones": self.dones[idx].to(self.device),
        }

    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    """On-policy rollout buffer for PPO with GAE advantage estimation.

    Parameters
    ----------
    state_dim : int
    max_size : int
        Maximum number of steps per rollout.
    device : str
    """

    def __init__(
        self,
        state_dim: int,
        max_size: int = 256,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.max_size = max_size
        self.device = device
        self._reset_storage()

    def _reset_storage(self) -> None:
        self.states = torch.zeros(self.max_size, self.state_dim)
        self.actions = torch.zeros(self.max_size, dtype=torch.long)
        self.rewards = torch.zeros(self.max_size)
        self.values = torch.zeros(self.max_size)
        self.log_probs = torch.zeros(self.max_size)
        self.dones = torch.zeros(self.max_size)
        self.advantages = torch.zeros(self.max_size)
        self.returns = torch.zeros(self.max_size)
        self.ptr = 0

    def add(self, transition: Transition) -> None:
        if self.ptr >= self.max_size:
            raise RuntimeError("RolloutBuffer is full. Call compute_returns() first.")
        i = self.ptr
        self.states[i] = torch.from_numpy(transition.state.astype(np.float32))
        self.actions[i] = int(transition.action)
        self.rewards[i] = float(transition.reward)
        self.values[i] = float(transition.value)
        self.log_probs[i] = float(transition.log_prob)
        self.dones[i] = float(transition.done)
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        next_value : float
            Bootstrapped value V(s_{T+1}).
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE-λ smoothing parameter.
        """
        last_gae = 0.0
        n = self.ptr
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_term = 1.0 - float(self.dones[t])
                next_val = next_value
            else:
                next_non_term = 1.0 - float(self.dones[t])
                next_val = float(self.values[t + 1])
            delta = (
                float(self.rewards[t])
                + gamma * next_val * next_non_term
                - float(self.values[t])
            )
            last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            self.advantages[t] = last_gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield shuffled mini-batches from the buffer."""
        n = self.ptr
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                "states": self.states[idx].to(self.device),
                "actions": self.actions[idx].to(self.device),
                "log_probs": self.log_probs[idx].to(self.device),
                "advantages": self.advantages[idx].to(self.device),
                "returns": self.returns[idx].to(self.device),
            }

    def clear(self) -> None:
        self._reset_storage()

    def is_full(self) -> bool:
        return self.ptr >= self.max_size

    def __len__(self) -> int:
        return self.ptr
