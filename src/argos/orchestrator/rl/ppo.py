# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Torch-backed Proximal Policy Optimization for discrete actions."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from argos.settings import RL_STATE_SIZE
from argos.orchestrator.rl.base import STANDARD_ACTIONS, RLAction, RLAgent, RLConfig
from argos.orchestrator.rl.torch_common import build_mlp, ensure_torch, torch_state_fingerprint, vectorize_state


@dataclass(frozen=True)
class PPOConfig(RLConfig):
    """Configuration for clipped discrete PPO."""

    hidden_size: int = 64
    state_size: int = RL_STATE_SIZE
    clip_epsilon: float = 0.2
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    gae_lambda: float = 0.95
    rollout_size: int = 32
    update_epochs: int = 4
    minibatch_size: int = 16
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    algorithm: str = "ppo"
    policy_version: str = "ppo-v3"


@dataclass(frozen=True)
class _RolloutTransition:
    state: list[float]
    action_idx: int
    reward: float
    value: float
    next_value: float
    old_log_prob: float
    done: bool
    allowed_indices: tuple[int, ...]


class _ActorCritic:
    """Actor and critic networks optimized through one parameter iterator."""

    def __init__(self, state_size: int, hidden_size: int, action_count: int):
        self.actor = build_mlp(state_size, hidden_size, action_count)
        self.critic = build_mlp(state_size, hidden_size, 1)

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])


class PPOAgent(RLAgent[Hashable, RLAction]):
    """Clipped PPO agent with rollout batching and generalized advantages."""

    def __init__(self, config: PPOConfig, actions: Optional[list[RLAction]] = None):
        super().__init__(config)
        torch, _ = ensure_torch()

        self.ppo_config = config
        self._actions = actions or STANDARD_ACTIONS
        self._num_actions = len(self._actions)
        self._torch = torch
        self._torch_generator = torch.Generator(device="cpu")
        if config.seed is None:
            self._torch_generator.seed()
            self._network = _ActorCritic(config.state_size, config.hidden_size, self._num_actions)
        else:
            self._torch_generator.manual_seed(config.seed)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.seed)
                self._network = _ActorCritic(config.state_size, config.hidden_size, self._num_actions)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=config.learning_rate)
        self._rollout: list[_RolloutTransition] = []
        self._pending_selection: Optional[tuple[list[float], int, float, float, tuple[int, ...]]] = None
        self._loss = 0.0
        self._actor_loss = 0.0
        self._critic_loss = 0.0
        self._entropy = 0.0
        self._update_count = 0
        self._last_was_exploration = False

        self.set_runtime_metadata(backend="torch")

    @property
    def policy_extension(self) -> str:
        return ".pt"

    @property
    def epsilon(self) -> float:
        """PPO samples from its policy and does not use epsilon-greedy exploration."""
        return 0.0

    @property
    def last_was_exploration(self) -> bool:
        return self._last_was_exploration

    def decay_exploration(self) -> None:
        """PPO exploration is controlled by policy entropy, not epsilon decay."""
        return None

    def _state_vector(self, state: Hashable) -> list[float]:
        return vectorize_state(state, self.ppo_config.state_size)

    def _state_tensor(self, state: Hashable):
        return self._torch.tensor(self._state_vector(state), dtype=self._torch.float32).unsqueeze(0)

    def select_action(
        self,
        state: Hashable,
        training: bool = True,
        allowed_action_indices: Optional[tuple[int, ...]] = None,
    ) -> RLAction:
        state_vector = self._state_vector(state)
        state_tensor = self._torch.tensor(state_vector, dtype=self._torch.float32).unsqueeze(0)
        allowed = tuple(sorted(set(allowed_action_indices or range(self._num_actions)))) or (0,)
        with self._torch.no_grad():
            logits = self._network.actor(state_tensor)
            masked_logits = self._torch.full_like(logits, float("-inf"))
            masked_logits[:, list(allowed)] = logits[:, list(allowed)]
            value = float(self._network.critic(state_tensor).squeeze().item())
            distribution = self._torch.distributions.Categorical(logits=masked_logits)
            if training:
                action_tensor = self._torch.multinomial(
                    distribution.probs,
                    num_samples=1,
                    generator=self._torch_generator,
                ).squeeze(1)
            else:
                action_tensor = self._torch.argmax(masked_logits, dim=1)
            action_idx = int(action_tensor.item())
            log_prob = float(distribution.log_prob(action_tensor).item())

        self._last_was_exploration = False
        self._pending_selection = (state_vector, action_idx, log_prob, value, allowed) if training else None
        return self._actions[action_idx]

    def _behavior_values(
        self,
        state: Hashable,
        action_idx: int,
        allowed_action_indices: Optional[tuple[int, ...]],
    ) -> tuple[list[float], float, float, tuple[int, ...]]:
        state_vector = self._state_vector(state)
        allowed = tuple(sorted(set(allowed_action_indices or range(self._num_actions)))) or (0,)
        pending = self._pending_selection
        if pending and pending[0] == state_vector and pending[1] == action_idx and pending[4] == allowed:
            self._pending_selection = None
            return pending[0], pending[2], pending[3], pending[4]
        self._pending_selection = None

        state_tensor = self._torch.tensor(state_vector, dtype=self._torch.float32).unsqueeze(0)
        action_tensor = self._torch.tensor([action_idx], dtype=self._torch.long)
        with self._torch.no_grad():
            logits = self._network.actor(state_tensor)
            masked_logits = self._torch.full_like(logits, float("-inf"))
            masked_logits[:, list(allowed)] = logits[:, list(allowed)]
            distribution = self._torch.distributions.Categorical(logits=masked_logits)
            log_prob = float(distribution.log_prob(action_tensor).item())
            value = float(self._network.critic(state_tensor).squeeze().item())
        return state_vector, log_prob, value, allowed

    def update(
        self,
        state: Hashable,
        action: RLAction,
        reward: float,
        next_state: Hashable,
        done: bool = False,
        allowed_action_indices: Optional[tuple[int, ...]] = None,
        next_allowed_action_indices: Optional[tuple[int, ...]] = None,
    ) -> dict[str, float]:
        action_idx = action.index if hasattr(action, "index") else self._actions.index(action)
        state_vector, old_log_prob, value, selected_allowed = self._behavior_values(
            state,
            action_idx,
            allowed_action_indices,
        )
        with self._torch.no_grad():
            next_value = 0.0 if done else float(self._network.critic(self._state_tensor(next_state)).squeeze().item())

        self._rollout.append(
            _RolloutTransition(
                state=state_vector,
                action_idx=action_idx,
                reward=float(reward),
                value=value,
                next_value=next_value,
                old_log_prob=old_log_prob,
                done=bool(done),
                allowed_indices=selected_allowed,
            )
        )
        self.step_complete(reward)

        optimized = done or len(self._rollout) >= max(1, self.ppo_config.rollout_size)
        if optimized:
            self._optimize_rollout()
        return {
            "loss": self._loss,
            "actor_loss": self._actor_loss,
            "critic_loss": self._critic_loss,
            "entropy": self._entropy,
            "reward": float(reward),
            "optimized": float(optimized),
        }

    def _advantages_and_returns(self) -> tuple[list[float], list[float]]:
        advantages = [0.0] * len(self._rollout)
        gae = 0.0
        gamma = self.ppo_config.discount_factor
        gae_lambda = self.ppo_config.gae_lambda
        for index in range(len(self._rollout) - 1, -1, -1):
            transition = self._rollout[index]
            continuation = 0.0 if transition.done else 1.0
            delta = transition.reward + gamma * transition.next_value * continuation - transition.value
            gae = delta + gamma * gae_lambda * continuation * gae
            advantages[index] = gae
        returns = [advantage + transition.value for advantage, transition in zip(advantages, self._rollout)]
        return advantages, returns

    def _optimize_rollout(self) -> None:
        if not self._rollout:
            return

        advantages, returns = self._advantages_and_returns()
        states = self._torch.tensor([row.state for row in self._rollout], dtype=self._torch.float32)
        actions = self._torch.tensor([row.action_idx for row in self._rollout], dtype=self._torch.long)
        old_log_probs = self._torch.tensor([row.old_log_prob for row in self._rollout], dtype=self._torch.float32)
        advantage_tensor = self._torch.tensor(advantages, dtype=self._torch.float32)
        return_tensor = self._torch.tensor(returns, dtype=self._torch.float32)
        allowed_mask = self._torch.zeros((len(self._rollout), self._num_actions), dtype=self._torch.bool)
        for row_index, row in enumerate(self._rollout):
            allowed_mask[row_index, list(row.allowed_indices)] = True

        if self.ppo_config.normalize_advantage and len(self._rollout) > 1:
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
                advantage_tensor.std(unbiased=False) + 1e-8
            )

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        losses: list[float] = []
        batch_size = max(1, min(self.ppo_config.minibatch_size, len(self._rollout)))

        for _ in range(max(1, self.ppo_config.update_epochs)):
            permutation = self._torch.randperm(
                len(self._rollout),
                generator=self._torch_generator,
            )
            for start in range(0, len(self._rollout), batch_size):
                indices = permutation[start : start + batch_size]
                logits = self._network.actor(states[indices])
                masked_logits = logits.masked_fill(~allowed_mask[indices], float("-inf"))
                distribution = self._torch.distributions.Categorical(logits=masked_logits)
                new_log_probs = distribution.log_prob(actions[indices])
                entropy = distribution.entropy().mean()
                values = self._network.critic(states[indices]).squeeze(-1)

                ratios = self._torch.exp(new_log_probs - old_log_probs[indices])
                unclipped = ratios * advantage_tensor[indices]
                clipped = self._torch.clamp(
                    ratios,
                    1.0 - self.ppo_config.clip_epsilon,
                    1.0 + self.ppo_config.clip_epsilon,
                ) * advantage_tensor[indices]
                actor_loss = -self._torch.min(unclipped, clipped).mean()
                critic_loss = self._torch.nn.functional.mse_loss(values, return_tensor[indices])
                loss = (
                    actor_loss
                    + self.ppo_config.value_weight * critic_loss
                    - self.ppo_config.entropy_weight * entropy
                )

                self._optimizer.zero_grad()
                loss.backward()
                self._torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.ppo_config.max_grad_norm)
                self._optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))
                losses.append(float(loss.item()))

        self._actor_loss = sum(actor_losses) / len(actor_losses)
        self._critic_loss = sum(critic_losses) / len(critic_losses)
        self._entropy = sum(entropies) / len(entropies)
        self._loss = sum(losses) / len(losses)
        self._update_count += 1
        self._rollout.clear()
        self._pending_selection = None

    def end_trajectory(self, final_state: Optional[Hashable] = None, done: bool = True) -> None:
        del final_state, done
        self._optimize_rollout()
        self._episode_count += 1

    def get_policy(self, state: Hashable) -> dict[RLAction, float]:
        with self._torch.no_grad():
            logits = self._network.actor(self._state_tensor(state))
            probs = self._torch.softmax(logits, dim=-1)[0].tolist()
        return {self._actions[i]: float(probs[i]) for i in range(self._num_actions)}

    def policy_fingerprint(self) -> str:
        """Hash actor and critic parameters without optimizer or run counters."""
        return torch_state_fingerprint(self._network.state_dict())

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "network": self._network.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "episode_count": self._episode_count,
                "step_count": self._step_count,
                "total_reward": self._total_reward,
                "update_count": self._update_count,
                "policy_metadata": self.get_policy_metadata(),
                "config": {
                    "algorithm": self.ppo_config.algorithm,
                    "policy_version": self.ppo_config.policy_version,
                    "state_size": self.ppo_config.state_size,
                    "hidden_size": self.ppo_config.hidden_size,
                    "learning_rate": self.ppo_config.learning_rate,
                    "discount_factor": self.ppo_config.discount_factor,
                    "clip_epsilon": self.ppo_config.clip_epsilon,
                    "entropy_weight": self.ppo_config.entropy_weight,
                    "value_weight": self.ppo_config.value_weight,
                    "rollout_size": self.ppo_config.rollout_size,
                    "update_epochs": self.ppo_config.update_epochs,
                    "minibatch_size": self.ppo_config.minibatch_size,
                    "gae_lambda": self.ppo_config.gae_lambda,
                },
            },
            save_path,
        )

    def load(self, path: str) -> None:
        checkpoint = self._torch.load(path, map_location="cpu")
        policy_metadata = checkpoint.get("policy_metadata", {})
        self.validate_policy_metadata(policy_metadata)
        self._network.load_state_dict(checkpoint["network"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._episode_count = int(checkpoint.get("episode_count", 0))
        self._step_count = int(checkpoint.get("step_count", 0))
        self._total_reward = float(checkpoint.get("total_reward", 0.0))
        self._update_count = int(checkpoint.get("update_count", 0))
        self._rollout.clear()
        self._pending_selection = None
        self.set_runtime_metadata(
            requested_algorithm=policy_metadata.get("requested_algorithm", self.ppo_config.algorithm),
            fallback_reason=policy_metadata.get("fallback_reason"),
            backend=policy_metadata.get("backend", "torch"),
        )

    def get_metrics(self) -> dict[str, Any]:
        metrics = super().get_metrics()
        metrics.update(
            {
                "epsilon": 0.0,
                "states_visited": self.step_count,
                "loss": self._loss,
                "actor_loss": self._actor_loss,
                "critic_loss": self._critic_loss,
                "entropy": self._entropy,
                "ppo_updates": self._update_count,
                "rollout_pending": len(self._rollout),
            }
        )
        return metrics
