"""Shared numeric contract for bounded analytics requirements."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional

CONTRACT_REL_TOL = 1e-9
CONTRACT_ABS_TOL = 1e-12
CONTRACT_DECIMALS = 12


def contract_tolerance(bound: float) -> float:
    """Return the tolerance used for one finite contract bound."""
    value = float(bound)
    if not math.isfinite(value):
        raise ValueError(f"Contract bound must be finite, got {bound!r}")
    return max(CONTRACT_ABS_TOL, CONTRACT_REL_TOL * max(abs(value), 1.0))


def breaches_lower_bound(value: float, bound: float) -> bool:
    """Return whether ``value`` is materially below ``bound``."""
    observed = float(value)
    if not math.isfinite(observed):
        return True
    return observed < float(bound) - contract_tolerance(bound)


def breaches_upper_bound(value: float, bound: float) -> bool:
    """Return whether ``value`` is materially above ``bound``."""
    observed = float(value)
    if not math.isfinite(observed):
        return True
    return observed > float(bound) + contract_tolerance(bound)


def within_contract_bounds(
    value: Optional[float],
    low: Optional[float],
    high: Optional[float],
) -> bool:
    """Return whether a finite value satisfies its optional closed bounds."""
    if value is None:
        return False
    observed = float(value)
    if not math.isfinite(observed):
        return False
    if low is not None and breaches_lower_bound(observed, low):
        return False
    return not (high is not None and breaches_upper_bound(observed, high))


def canonical_contract_value(
    value: float,
    bounds: Optional[tuple[float, float]] = None,
) -> float:
    """Return a path-independent representation of a contract value.

    Requirement actions use decimal step sizes that are not exactly
    representable as binary floats. Rounding to a precision far below the
    action granularity removes accumulated ULP drift without moving genuine
    operating points such as a midpoint at 0.325.
    """
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError(f"Contract value must be finite, got {value!r}")

    if bounds is not None:
        low, high = map(float, bounds)
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"Invalid contract bounds: {bounds!r}")
        if not breaches_lower_bound(observed, low) and observed < low:
            observed = low
        if not breaches_upper_bound(observed, high) and observed > high:
            observed = high
        observed = min(max(observed, low), high)

    canonical = round(observed, CONTRACT_DECIMALS)
    if bounds is not None:
        low, high = map(float, bounds)
        if math.isclose(canonical, low, rel_tol=CONTRACT_REL_TOL, abs_tol=CONTRACT_ABS_TOL):
            canonical = low
        elif math.isclose(canonical, high, rel_tol=CONTRACT_REL_TOL, abs_tol=CONTRACT_ABS_TOL):
            canonical = high
    return 0.0 if canonical == 0.0 else canonical


def canonical_contract_mean(
    values: Iterable[float],
    bounds: Optional[tuple[float, float]] = None,
) -> Optional[float]:
    """Return a canonical arithmetic mean, or ``None`` for an empty input."""
    observed = [float(value) for value in values]
    if not observed:
        return None
    if any(not math.isfinite(value) for value in observed):
        raise ValueError("Contract mean requires finite values")
    return canonical_observed_value(math.fsum(observed) / len(observed), bounds)


def canonical_observed_value(
    value: float,
    bounds: Optional[tuple[float, float]] = None,
) -> float:
    """Canonicalize an observation without hiding a material contract breach."""
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError(f"Observed contract value must be finite, got {value!r}")
    canonical = round(observed, CONTRACT_DECIMALS)
    if bounds is not None:
        low, high = map(float, bounds)
        if math.isclose(canonical, low, rel_tol=CONTRACT_REL_TOL, abs_tol=CONTRACT_ABS_TOL):
            canonical = low
        elif math.isclose(canonical, high, rel_tol=CONTRACT_REL_TOL, abs_tol=CONTRACT_ABS_TOL):
            canonical = high
    return 0.0 if canonical == 0.0 else canonical


def contract_position(value: float, bounds: tuple[float, float]) -> float:
    """Map a bounded contract value to a stable relative position in [0, 1]."""
    low, high = map(float, bounds)
    canonical = canonical_contract_value(value, (low, high))
    width = high - low
    if width <= CONTRACT_ABS_TOL:
        return 0.0
    return canonical_contract_value((canonical - low) / width, (0.0, 1.0))
