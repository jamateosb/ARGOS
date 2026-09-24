# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Optional torch-backed Deep Q-Network agent."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from argos.settings import RL_STATE_SIZE
from argos.orchestrator.rl.base import STANDARD_ACTIONS, RLAction, RLAgent, RLConfig
from argos.orchestrator.rl.torch_common import build_mlp, ensure_torch, torch_state_fingerprint, vectorize_state


@dataclass(frozen=True)
class DQNConfig(RLConfig):
    """Configuration for the compact DQN backend."""

    hidden_size: int = 64
    state_size: int = RL_STATE_SIZE
    target_update_interval: int = 25
    replay_warmup: int = 32
    algorithm: str = "dqn"
    max_grad_norm: float = 1.0
    policy_version: str = "dqn-v3"


class DQNAgent(RLAgent[Hashable, RLAction]):
    """Minimal DQN agent with replay buffer and target network."""

    def __init__(self, config: DQNConfig, actions: Optional[list[RLAction]] = None):
        super().__init__(config)
        torch, _ = ensure_torch()

        self.dqn_config = config
        self._actions = actions or STANDARD_ACTIONS
        self._num_actions = len(self._actions)
        self._memory: deque = deque(maxlen=config.memory_size)
        self._loss: float = 0.0
        self._last_was_exploration = False
        self._torch = torch

        self._rng = random.Random(config.seed)
        if config.seed is None:
            self._q_network = build_mlp(config.state_size, config.hidden_size, self._num_actions)
            self._target_network = build_mlp(config.state_size, config.hidden_size, self._num_actions)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.seed)
                self._q_network = build_mlp(config.state_size, config.hidden_size, self._num_actions)
                self._target_network = build_mlp(config.state_size, config.hidden_size, self._num_actions)
        self._target_network.load_state_dict(self._q_network.state_dict())
        self._optimizer = torch.optim.Adam(self._q_network.parameters(), lr=config.learning_rate)
        self._loss_fn = torch.nn.SmoothL1Loss()
        self.set_runtime_metadata(backend="torch")

    @property
    def policy_extension(self) -> str:
        return ".pt"

    @property
    def epsilon(self) -> float:
        return self._exploration_rate

    @property
    def last_was_exploration(self) -> bool:
        return self._last_was_exploration

    def _state_tensor(self, state: Hashable):
        vector = vectorize_state(state, self.dqn_config.state_size)
        return self._torch.tensor(vector, dtype=self._torch.float32).unsqueeze(0)

    def select_action(
        self,
        state: Hashable,
        training: bool = True,
        allowed_action_indices: Optional[tuple[int, ...]] = None,
    ) -> RLAction:
        allowed = tuple(sorted(set(allowed_action_indices or range(self._num_actions)))) or (0,)
        if training and self._rng.random() < self._exploration_rate:
            self._last_was_exploration = True
            return self._actions[self._rng.choice(allowed)]

        self._last_was_exploration = False
        with self._torch.no_grad():
            q_values = self._q_network(self._state_tensor(state))[0]
            masked_q = self._torch.full_like(q_values, float("-inf"))
            masked_q[list(allowed)] = q_values[list(allowed)]
            action_idx = int(self._torch.argmax(masked_q).item())
        return self._actions[action_idx]

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
        state_vec = vectorize_state(state, self.dqn_config.state_size)
        next_state_vec = vectorize_state(next_state, self.dqn_config.state_size)
        next_allowed = tuple(sorted(set(next_allowed_action_indices or range(self._num_actions)))) or (0,)
        self._memory.append((state_vec, action_idx, reward, next_state_vec, done, next_allowed))

        minimum_replay = max(self.dqn_config.batch_size, self.dqn_config.replay_warmup)
        if len(self._memory) < minimum_replay:
            self.step_complete(reward)
            return {"loss": 0.0, "reward": reward}

        batch_size = self.dqn_config.batch_size
        batch = self._rng.sample(list(self._memory), batch_size)
        states = self._torch.tensor([row[0] for row in batch], dtype=self._torch.float32)
        actions = self._torch.tensor([row[1] for row in batch], dtype=self._torch.long)
        rewards = self._torch.tensor([row[2] for row in batch], dtype=self._torch.float32)
        next_states = self._torch.tensor([row[3] for row in batch], dtype=self._torch.float32)
        dones = self._torch.tensor([row[4] for row in batch], dtype=self._torch.float32)

        current_q = self._q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with self._torch.no_grad():
            next_values = self._target_network(next_states)
            allowed_mask = self._torch.zeros_like(next_values, dtype=self._torch.bool)
            for row_index, row in enumerate(batch):
                allowed_mask[row_index, list(row[5])] = True
            next_q = next_values.masked_fill(~allowed_mask, float("-inf")).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.dqn_config.discount_factor * next_q

        loss = self._loss_fn(current_q, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._torch.nn.utils.clip_grad_norm_(self._q_network.parameters(), self.dqn_config.max_grad_norm)
        self._optimizer.step()

        self._loss = float(loss.item())
        self.step_complete(reward)

        if self.step_count % max(1, self.dqn_config.target_update_interval) == 0:
            self._target_network.load_state_dict(self._q_network.state_dict())

        return {"loss": self._loss, "reward": reward}

    def get_policy(self, state: Hashable) -> dict[RLAction, float]:
        with self._torch.no_grad():
            q_values = self._q_network(self._state_tensor(state))[0].tolist()
        return {self._actions[i]: float(q_values[i]) for i in range(self._num_actions)}

    def policy_fingerprint(self) -> str:
        """Hash the online value network that determines greedy actions."""
        return torch_state_fingerprint(self._q_network.state_dict())

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "q_network": self._q_network.state_dict(),
                "target_network": self._target_network.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "episode_count": self._episode_count,
                "step_count": self._step_count,
                "total_reward": self._total_reward,
                "exploration_rate": self._exploration_rate,
                "policy_metadata": self.get_policy_metadata(),
                "config": {
                    "algorithm": self.dqn_config.algorithm,
                    "policy_version": self.dqn_config.policy_version,
                    "state_size": self.dqn_config.state_size,
                    "hidden_size": self.dqn_config.hidden_size,
                    "learning_rate": self.dqn_config.learning_rate,
                    "discount_factor": self.dqn_config.discount_factor,
                    "batch_size": self.dqn_config.batch_size,
                    "replay_warmup": self.dqn_config.replay_warmup,
                    "target_update_interval": self.dqn_config.target_update_interval,
                },
            },
            save_path,
        )

    def load(self, path: str) -> None:
        checkpoint = self._torch.load(path, map_location="cpu")
        policy_metadata = checkpoint.get("policy_metadata", {})
        self.validate_policy_metadata(policy_metadata)
        self._q_network.load_state_dict(checkpoint["q_network"])
        self._target_network.load_state_dict(checkpoint["target_network"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._episode_count = int(checkpoint.get("episode_count", 0))
        self._step_count = int(checkpoint.get("step_count", 0))
        self._total_reward = float(checkpoint.get("total_reward", 0.0))
        self._exploration_rate = float(checkpoint.get("exploration_rate", self.config.exploration_rate))
        self.set_runtime_metadata(
            requested_algorithm=policy_metadata.get("requested_algorithm", self.dqn_config.algorithm),
            fallback_reason=policy_metadata.get("fallback_reason"),
            backend=policy_metadata.get("backend", "torch"),
        )

    def get_metrics(self) -> dict[str, Any]:
        metrics = super().get_metrics()
        metrics.update(
            {
                "epsilon": self._exploration_rate,
                "states_visited": len(self._memory),
                "loss": self._loss,
                "replay_size": len(self._memory),
            }
        )
        return metrics
