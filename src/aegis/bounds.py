"""Theory-aligned decision bounds for AEGIS."""

from __future__ import annotations


def _validate_nonnegative(value: float, *, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _utility_span(utility_range: tuple[float, float]) -> float:
    lower, upper = float(utility_range[0]), float(utility_range[1])
    if upper < lower:
        raise ValueError("utility_range upper bound must be >= lower bound.")
    return upper - lower


def negative_control_sensitivity_bound(delta_nc: float, kappa: float = 1.0) -> float:
    """Return the observable upper bound delta <= kappa * delta_nc."""
    delta_nc = _validate_nonnegative(delta_nc, name="delta_nc")
    kappa = _validate_nonnegative(kappa, name="kappa")
    return kappa * delta_nc


def policy_regret_upper_bound(
    regret_observational: float,
    delta_uniform: float,
    utility_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Bound interventional regret via observational regret + M * delta_uniform."""
    regret_observational = float(regret_observational)
    delta_uniform = _validate_nonnegative(delta_uniform, name="delta_uniform")
    m = _utility_span(utility_range)
    return regret_observational + m * delta_uniform


def policy_regret_minimax_floor(
    delta_uniform: float,
    utility_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Return the minimax floor 0.5 * M * delta_uniform."""
    delta_uniform = _validate_nonnegative(delta_uniform, name="delta_uniform")
    m = _utility_span(utility_range)
    return 0.5 * m * delta_uniform

