"""Proximal Policy Optimisation (PPO) — self-contained implementation.

No Stable-Baselines3 dependency. Clean internal interface so the RL
algorithm is not tightly coupled to any external framework.

The PPO update is:
    L_CLIP = E[ min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t) ]
    L_VALUE = (V(s_t) - R_t)²
    L_ENTROPY = H[π(·|s_t)]
    L = -L_CLIP + c_v L_VALUE - c_e L_ENTROPY

The RL policy receives ONLY RL gradients — the PINN physics loss does
NOT flow through the policy network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from module_08_pinn_rl.config import RLConfig
from module_08_pinn_rl.environment import SyntheticKinematicEnv
from module_08_pinn_rl.replay import RolloutBuffer
from module_08_pinn_rl.rl_policy import MLPPolicy
from module_08_pinn_rl.transitions import Transition


class PPO:
    """Lightweight self-contained PPO implementation.

    Parameters
    ----------
    policy : MLPPolicy
        Actor-Critic policy network.
    config : RLConfig
        Hyperparameters.
    device : str
    """

    def __init__(
        self,
        policy: MLPPolicy,
        config: RLConfig,
        device: str = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.buffer = RolloutBuffer(
            state_dim=policy.actor[0].in_features
            if hasattr(policy.actor[0], "in_features")
            else list(policy.actor.parameters())[0].shape[1],
            max_size=config.n_steps * 2,
            device=device,
        )

    def collect_rollout(
        self,
        env: SyntheticKinematicEnv,
        dynamics_model=None,
    ) -> Tuple[float, int]:
        """Collect n_steps of experience.

        Parameters
        ----------
        env : SyntheticKinematicEnv (or compatible)
        dynamics_model : PhysicsInformedDynamicsModel, optional
            If provided, uses PINN to predict next states (Exp C).
            If None, uses real environment dynamics (Exp A).

        Returns
        -------
        (mean_episode_reward, n_episodes_completed)
        """
        self.policy.eval()
        self.buffer.clear()

        obs, _ = env.reset()
        episode_rewards: List[float] = []
        ep_reward = 0.0
        n_episodes = 0

        for _ in range(self.config.n_steps):
            state_t = torch.from_numpy(obs.astype(np.float32)).to(self.device)

            with torch.no_grad():
                action, log_prob, _, value = self.policy.get_action_and_value(state_t)

            action_int = int(action.item())

            if dynamics_model is not None:
                # RL + PINN path (Exp C): use PINN for next-state prediction
                from module_08_pinn_rl.action import ActionEncoder, ActionSpec
                spec = ActionSpec(
                    action_type=self.config.action_type,
                    action_dim=self.config.action_dim,
                )
                enc = ActionEncoder(spec)
                encoded = enc.encode(action_int)
                next_obs = dynamics_model.predict(obs, action_int)
                next_obs = next_obs.astype(np.float32)
                # Get reward from environment without advancing its state
                _, reward, terminated, truncated, info = env.step(action_int)
                done = terminated or truncated
            else:
                # RL-only path (Exp A): use real environment dynamics
                next_obs, reward, terminated, truncated, info = env.step(action_int)
                done = terminated or truncated

            ep_reward += reward

            trans = Transition(
                state=obs,
                action=action_int,
                reward=reward,
                next_state=next_obs,
                done=done,
                log_prob=float(log_prob.item()),
                value=float(value.item()),
            )
            self.buffer.add(trans)
            obs = next_obs

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                n_episodes += 1
                obs, _ = env.reset()

        # Bootstrap
        with torch.no_grad():
            last_state = torch.from_numpy(obs.astype(np.float32)).to(self.device)
            _, _, _, last_value = self.policy.get_action_and_value(last_state)
        self.buffer.compute_returns_and_advantages(
            next_value=float(last_value.item()),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else ep_reward
        return mean_reward, n_episodes

    def update(self) -> Dict[str, float]:
        """Run PPO mini-batch updates over the collected rollout.

        Returns
        -------
        dict of mean losses over all epochs and mini-batches.
        """
        self.policy.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalize advantages
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                _, new_log_probs, entropy, new_values = (
                    self.policy.get_action_and_value(states, actions)
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((new_values - returns) ** 2).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
        }

    def train(
        self,
        env: SyntheticKinematicEnv,
        n_rollouts: int = 10,
        dynamics_model=None,
    ) -> List[Dict[str, float]]:
        """Full PPO training loop.

        Parameters
        ----------
        env : environment
        n_rollouts : int
            Number of rollout → update cycles.
        dynamics_model : optional
            PINN dynamics model for Exp C (RL + PINN).

        Returns
        -------
        List of per-rollout metric dicts.
        """
        history = []
        for rollout_idx in range(n_rollouts):
            mean_reward, n_eps = self.collect_rollout(env, dynamics_model)
            losses = self.update()
            record = {
                "rollout": rollout_idx,
                "mean_reward": mean_reward,
                "n_episodes": n_eps,
                **losses,
            }
            history.append(record)
        return history
