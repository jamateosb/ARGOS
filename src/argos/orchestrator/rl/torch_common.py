# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Shared helpers for optional torch-backed RL agents."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Mapping

from argos.orchestrator.rl.base import RLState

try:  # pragma: no cover - import availability is environment-dependent
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - handled via factory fallback
    torch = None
    nn = None


def torch_available() -> bool:
    """Return True when the optional torch backend is importable."""
    return torch is not None and nn is not None


def ensure_torch():
    """Raise a clear error when optional torch extras are unavailable."""
    if not torch_available():
        raise ImportError(
            "PyTorch is required for DQN/PPO backends. " "Install the optional RL extra or fall back to qlearning."
        )
    return torch, nn


def vectorize_state(state: Hashable, state_size: int) -> list[float]:
    """Convert a hashable RL state into a fixed-width float vector."""
    if isinstance(state, RLState):
        values = [float(v) for v in state.to_vector()]
    elif isinstance(state, (list, tuple)):
        values = [float(v) for v in state]
    else:
        values = [float(hash(state) % 10_000) / 10_000.0]

    if len(values) < state_size:
        values.extend([0.0] * (state_size - len(values)))
    return values[:state_size]


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    """Build a compact MLP used by DQN and PPO backends."""
    _, nn_mod = ensure_torch()
    return nn_mod.Sequential(
        nn_mod.Linear(input_dim, hidden_dim),
        nn_mod.ReLU(),
        nn_mod.Linear(hidden_dim, hidden_dim),
        nn_mod.ReLU(),
        nn_mod.Linear(hidden_dim, output_dim),
    )


def torch_state_fingerprint(state: Mapping) -> str:
    """Hash nested Torch state dictionaries without optimizer metadata."""
    digest = hashlib.sha256()

    def _update(prefix: str, value) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                _update(f"{prefix}.{key}", value[key])
            return
        tensor = value.detach().cpu().contiguous()
        digest.update(prefix.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())

    _update("policy", state)
    return digest.hexdigest()
