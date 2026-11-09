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
    n_actions: int = 2,
    policy_type: str = "covariate",
) -> float:
    """Bound interventional regret: simulated regret + 2 * k * M * delta.

    ``regret_observational`` is the regret computed under the adjusted (simulated)
    interventional laws. ``k = n_actions`` for covariate-dependent policies and
    ``k = 1`` when comparing fixed actions (``policy_type="constant"``).
    Versions up to 0.1.0 used ``M * delta``, which does not bound the regret
    (manuscript, Policy Regret Transfer theorem; the factor 2 is necessary).
    """
    regret_observational = float(regret_observational)
    delta_uniform = _validate_nonnegative(delta_uniform, name="delta_uniform")
    m = _utility_span(utility_range)
    if int(n_actions) < 2:
        raise ValueError("n_actions must be at least 2.")
    if policy_type not in ("covariate", "constant"):
        raise ValueError("policy_type must be 'covariate' or 'constant'.")
    k = int(n_actions) if policy_type == "covariate" else 1
    return regret_observational + 2.0 * k * m * delta_uniform


def policy_regret_minimax_floor(
    delta_uniform: float,
    utility_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Return the minimax floor 0.25 * M * delta_uniform.

    The constant 1/4 is sharp (manuscript, Minimax Safety Floor theorem). Versions
    up to 0.1.0 returned 0.5 * M * delta_uniform, which is false.
    """
    delta_uniform = _validate_nonnegative(delta_uniform, name="delta_uniform")
    m = _utility_span(utility_range)
    return 0.25 * m * delta_uniform

