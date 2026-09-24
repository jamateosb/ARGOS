# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Benchmark-driven RL policy selection utilities."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from argos.common.constants import REPO_ROOT
from argos.domain.requests import SUPPORTED_RL_ALGORITHMS, normalize_algorithm_name

AUTO_ALGORITHM_ALIASES = {"auto", "best", "benchmark", "benchmarkbest", "benchmarkwinner"}
DEFAULT_BENCHMARK_RANKING = Path("data/evaluation/comparisons/mixed_campaign/latest/model_ranking_ci.csv")


@dataclass(frozen=True)
class BenchmarkPolicySelection:
    """Resolved algorithm plus the benchmark row that selected it."""

    algorithm: str
    runtime: Optional[str] = None
    mode: Optional[str] = None
    reward_mean: Optional[float] = None
    source_path: Optional[Path] = None
    fallback_reason: Optional[str] = None


def is_auto_algorithm(value: Optional[str]) -> bool:
    """Return True when a public algorithm value asks for benchmark-based selection."""
    raw = (value or "auto").strip().lower().replace("-", "").replace("_", "")
    return raw in AUTO_ALGORITHM_ALIASES


def resolve_algorithm(
    value: Optional[str],
    *,
    config: Optional[dict[str, Any]] = None,
    fallback_algorithm: str = "qlearning",
) -> str:
    """Resolve public algorithm input to a concrete supported backend."""
    if is_auto_algorithm(value):
        return select_best_policy_from_benchmark(config=config, fallback_algorithm=fallback_algorithm).algorithm

    normalized = normalize_algorithm_name(value or fallback_algorithm)
    if normalized not in SUPPORTED_RL_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm '{value}'. Expected auto or one of {SUPPORTED_RL_ALGORITHMS}")
    return normalized


def select_best_policy_from_benchmark(
    *,
    config: Optional[dict[str, Any]] = None,
    fallback_algorithm: str = "qlearning",
) -> BenchmarkPolicySelection:
    """
    Select the best concrete RL algorithm from the latest benchmark ranking.

    The ranking CSV is expected to be sorted by the evidence generator, but this
    function also compares ``reward_mean`` so a regenerated unsorted file still
    produces the best available row. Aggregate ``runtime=all`` rows are ignored
    when a concrete runtime row is available, because deployment must run in one
    actual runtime mode.
    """
    fallback = normalize_algorithm_name(fallback_algorithm)
    if fallback not in SUPPORTED_RL_ALGORITHMS:
        fallback = "qlearning"

    for path in _ranking_candidates(config):
        if not path.exists():
            continue
        selection = _selection_from_csv(path)
        if selection is not None:
            return selection

    return BenchmarkPolicySelection(
        algorithm=fallback,
        fallback_reason="No usable benchmark ranking CSV found; using fallback algorithm",
    )


def _selection_from_csv(path: Path) -> Optional[BenchmarkPolicySelection]:
    rows: list[tuple[int, BenchmarkPolicySelection]] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                algorithm = normalize_algorithm_name(str(row.get("algorithm") or ""))
                if algorithm not in SUPPORTED_RL_ALGORITHMS:
                    continue
                # The static baseline never participates in the auto-selection
                # ranking: 'auto' is meant to choose the best learning agent,
                # not the no-op baseline used as a control.
                if algorithm in {"static", "threshold"}:
                    continue
                rows.append(
                    (
                        index,
                        BenchmarkPolicySelection(
                            algorithm=algorithm,
                            runtime=_clean_optional(row.get("runtime")),
                            mode=_clean_optional(row.get("mode")),
                            reward_mean=_float_or_none(row.get("reward_mean")),
                            source_path=path,
                        ),
                    )
                )
    except OSError:
        return None

    if not rows:
        return None

    concrete_runtime_rows = [(idx, item) for idx, item in rows if (item.runtime or "").lower() != "all"]
    candidates = concrete_runtime_rows or rows
    _index, best = max(candidates, key=lambda item: (_score(item[1].reward_mean), -item[0]))
    return best


def _ranking_candidates(config: Optional[dict[str, Any]]) -> list[Path]:
    root = REPO_ROOT
    candidates: list[Path] = []

    env_path = os.getenv("ARGOS_BENCHMARK_RANKING_CSV")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    paths_cfg = (config or {}).get("paths", {}) if isinstance(config, dict) else {}
    runtime_cfg = (config or {}).get("runtime", {}) if isinstance(config, dict) else {}
    evaluation_dir = (
        paths_cfg.get("evaluation_dir")
        or paths_cfg.get("evaluation_output_dir")
        or runtime_cfg.get("evaluation_output_dir")
    )
    if evaluation_dir:
        candidates.append(
            _absolute_path(Path(str(evaluation_dir)).expanduser(), root)
            / "comparisons/mixed_campaign/latest/model_ranking_ci.csv"
        )

    candidates.extend(
        [
            root / DEFAULT_BENCHMARK_RANKING,
            Path.cwd() / DEFAULT_BENCHMARK_RANKING,
            Path("/opt/argos/ARGOS") / DEFAULT_BENCHMARK_RANKING,
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _absolute_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _clean_optional(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _score(value: Optional[float]) -> float:
    return value if value is not None else float("-inf")
