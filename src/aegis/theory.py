"""Theory-aligned diagnostics inspired by the causaldef R package."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .monitor import deficiency_proxy_propensity


def rkhs_rate_bound(
    n: int,
    beta: float,
    d_w: int,
    eta: float,
    xi: float = 0.05,
) -> float:
    """Finite-sample RKHS rate bound surrogate."""
    n = int(n)
    d_w = int(d_w)
    beta = float(beta)
    eta = float(eta)
    xi = float(xi)
    if n <= 0:
        raise ValueError("n must be positive.")
    if d_w <= 0:
        raise ValueError("d_w must be positive.")
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if eta <= 0:
        return float("inf")
    if xi <= 0:
        xi = np.finfo(float).eps
    if xi >= 1:
        raise ValueError("xi must be in (0, 1).")

    return float(n ** (-beta / (2.0 * beta + d_w)) + (1.0 / eta) * np.sqrt(np.log(1.0 / xi) / n))


def policy_regret_vc_bound(
    regret_observational: float,
    delta_uniform: float,
    vc_dim: float,
    n: int,
    utility_range: Tuple[float, float] = (0.0, 1.0),
    xi: float = 0.05,
    c: float = 2.0,
) -> dict:
    """Policy regret upper bound with VC complexity term."""
    regret_observational = float(regret_observational)
    delta_uniform = float(delta_uniform)
    vc_dim = float(vc_dim)
    n = int(n)
    c = float(c)
    if delta_uniform < 0:
        raise ValueError("delta_uniform must be non-negative.")
    if vc_dim < 0:
        raise ValueError("vc_dim must be non-negative.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if c < 0:
        raise ValueError("c must be non-negative.")
    if xi <= 0:
        xi = np.finfo(float).eps
    if xi >= 1:
        raise ValueError("xi must be in (0, 1).")

    lo, hi = float(utility_range[0]), float(utility_range[1])
    if hi < lo:
        raise ValueError("utility_range must satisfy max >= min.")
    m = hi - lo

    transfer_penalty = m * delta_uniform
    complexity_penalty = c * m * np.sqrt((vc_dim * np.log(n) + np.log(1.0 / xi)) / n)
    return {
        "regret_upper_bound": regret_observational + transfer_penalty + complexity_penalty,
        "transfer_penalty": transfer_penalty,
        "complexity_penalty": float(complexity_penalty),
        "delta_uniform": delta_uniform,
        "vc_dim": vc_dim,
        "n": n,
    }


def wasserstein_deficiency_gaussian(
    alpha: float,
    gamma: float,
    sigma_a: float = 1.0,
    a: float = 1.0,
) -> float:
    """Closed-form W1 deficiency for the linear Gaussian two-point setting."""
    alpha = float(alpha)
    gamma = float(gamma)
    sigma_a = float(sigma_a)
    a = float(a)
    if sigma_a < 0:
        raise ValueError("sigma_a must be non-negative.")
    denom = alpha * alpha + sigma_a * sigma_a
    if denom <= 0:
        return 0.0
    return float(abs(a * alpha * gamma) / denom)


def _tv_distance_normal(mu1: float, sd1: float, mu2: float, sd2: float) -> float:
    """Total variation distance between two univariate normals.

    Computed via piecewise Simpson's-rule integration of |f1 - f2| over
    breakpoints scaled to each component's own sd (+/-5, 15, 30 sd around
    each mean). A closed-form quadratic-crossing-point approach was used
    here previously; it loses precision and silently returns near-0
    instead of near-1 when both sd's are very small relative to the mean
    separation (a near-degenerate point-mass regime). This mirrors the
    fix applied to causaldef's `.tv_distance_normal()` (2026-08-10),
    verified against independent quadrature to ~5e-7 over sd spanning
    1e-6 to 1e3.
    """
    if sd1 <= 0 or sd2 <= 0:
        return float("nan")
    if mu1 == mu2 and sd1 == sd2:
        return 0.0

    breakpoints = sorted(
        {
            mu1 + k * sd1 for k in (-30, -15, -5, 0, 5, 15, 30)
        }
        | {
            mu2 + k * sd2 for k in (-30, -15, -5, 0, 5, 15, 30)
        }
    )

    def f(x: np.ndarray) -> np.ndarray:
        f1 = np.exp(-0.5 * ((x - mu1) / sd1) ** 2) / (sd1 * np.sqrt(2.0 * np.pi))
        f2 = np.exp(-0.5 * ((x - mu2) / sd2) ** 2) / (sd2 * np.sqrt(2.0 * np.pi))
        return np.abs(f1 - f2)

    total = 0.0
    n_points = 2001  # odd -> even number of Simpson sub-intervals
    for lo, hi in zip(breakpoints[:-1], breakpoints[1:]):
        if hi <= lo:
            continue
        xs = np.linspace(lo, hi, n_points)
        ys = f(xs)
        h = (hi - lo) / (n_points - 1)
        total += h / 3.0 * (ys[0] + ys[-1] + 4.0 * np.sum(ys[1:-1:2]) + 2.0 * np.sum(ys[2:-2:2]))

    return float(np.clip(0.5 * total, 0.0, 1.0))


def _deficiency_gaussian(alpha: float, gamma: float, sigma_a: float, sigma_y: float, a: float) -> float:
    denom_a = alpha * alpha + sigma_a * sigma_a
    if denom_a <= 0:
        return 0.0
    delta_mu = abs(a * alpha * gamma) / denom_a
    gamma_prime_sq = (gamma * gamma) * (sigma_a * sigma_a) / denom_a

    sd1 = np.sqrt(gamma * gamma + sigma_y * sigma_y)
    sd2 = np.sqrt(gamma_prime_sq + sigma_y * sigma_y)
    tv = _tv_distance_normal(0.0, sd1, delta_mu, sd2)
    if not np.isfinite(tv):
        return float("nan")
    return float(0.5 * tv)


def sharp_lower_bound(
    alpha: float,
    gamma: float,
    sigma_a: float = 1.0,
    sigma_y: float = 1.0,
    a: float = 1.0,
    metric: str = "tv",
) -> dict:
    """Two-point sharp lower/upper bound helper."""
    metric = str(metric).lower().strip()
    if metric not in {"tv", "wasserstein"}:
        raise ValueError("metric must be one of {'tv', 'wasserstein'}.")
    if sigma_a < 0 or sigma_y < 0:
        raise ValueError("sigma_a and sigma_y must be non-negative.")

    if metric == "wasserstein":
        delta = wasserstein_deficiency_gaussian(alpha=alpha, gamma=gamma, sigma_a=sigma_a, a=a)
    else:
        delta = _deficiency_gaussian(
            alpha=float(alpha),
            gamma=float(gamma),
            sigma_a=float(sigma_a),
            sigma_y=float(sigma_y),
            a=float(a),
        )
    return {"metric": metric, "lower": delta, "upper": delta, "ratio": 1.0}


def partial_id_set(
    estimate: float,
    delta: float,
    estimand: str = "ate",
    outcome_range: Tuple[float, float] = (0.0, 1.0),
) -> dict:
    """Convert a deficiency radius to a conservative identification interval."""
    estimand = str(estimand).lower().strip()
    if estimand not in {"mean", "ate"}:
        raise ValueError("estimand must be one of {'mean', 'ate'}.")
    estimate = float(estimate)
    delta = float(delta)
    if delta < 0:
        raise ValueError("delta must be non-negative.")
    lo, hi = float(outcome_range[0]), float(outcome_range[1])
    if hi <= lo:
        raise ValueError("outcome_range must satisfy max > min.")
    span = hi - lo
    factor = 2.0 if estimand == "mean" else 4.0
    half_width = factor * delta * span
    return {
        "estimand": estimand,
        "estimate": estimate,
        "delta": delta,
        "outcome_range": (lo, hi),
        "half_width": half_width,
        "lower": estimate - half_width,
        "upper": estimate + half_width,
    }


def overlap_diagnostic(
    x: np.ndarray,
    a: np.ndarray,
    trim: float = 0.01,
    l2: float = 1e-3,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> dict:
    """Overlap/positivity summary from propensity scores and IPTW ESS."""
    trim = float(trim)
    if trim < 0 or trim >= 0.5:
        raise ValueError("trim must be in [0, 0.5).")

    delta, e = deficiency_proxy_propensity(
        x,
        a,
        metric="tv",
        l2=l2,
        max_iter=max_iter,
        tol=tol,
    )
    a_bin = np.asarray(a).reshape(-1).astype(float)
    e = np.clip(e, 1e-6, 1.0 - 1e-6)
    w = a_bin / e + (1.0 - a_bin) / (1.0 - e)
    ess = float((np.sum(w) ** 2) / np.sum(w * w))

    quantiles = np.quantile(e, [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "trim": trim,
        "n": int(a_bin.size),
        "extreme_n": int(np.sum((e < trim) | (e > (1.0 - trim)))),
        "kept_n": int(np.sum((e >= trim) & (e <= (1.0 - trim)))),
        "ess_iptw": ess,
        "delta_propensity": float(delta),
        "propensity_quantiles": {
            "p00": float(quantiles[0]),
            "p01": float(quantiles[1]),
            "p05": float(quantiles[2]),
            "p10": float(quantiles[3]),
            "p50": float(quantiles[4]),
            "p90": float(quantiles[5]),
            "p95": float(quantiles[6]),
            "p99": float(quantiles[7]),
            "p100": float(quantiles[8]),
        },
    }


def confounding_frontier(
    alpha_range: Tuple[float, float] = (-2.0, 2.0),
    gamma_range: Tuple[float, float] = (-2.0, 2.0),
    grid_size: int = 25,
    sigma_a: float = 1.0,
    sigma_y: float = 1.0,
    a: float = 1.0,
) -> dict:
    """Map a linear-Gaussian confounding frontier via two-point TV bounds."""
    if grid_size < 5:
        raise ValueError("grid_size must be >= 5.")
    if sigma_a < 0 or sigma_y < 0:
        raise ValueError("sigma_a and sigma_y must be non-negative.")

    alpha_seq = np.linspace(float(alpha_range[0]), float(alpha_range[1]), int(grid_size))
    gamma_seq = np.linspace(float(gamma_range[0]), float(gamma_range[1]), int(grid_size))

    alpha_grid, gamma_grid = np.meshgrid(alpha_seq, gamma_seq, indexing="xy")
    delta_grid = np.zeros_like(alpha_grid, dtype=float)
    for i in range(delta_grid.shape[0]):
        for j in range(delta_grid.shape[1]):
            delta_grid[i, j] = _deficiency_gaussian(
                alpha=float(alpha_grid[i, j]),
                gamma=float(gamma_grid[i, j]),
                sigma_a=float(sigma_a),
                sigma_y=float(sigma_y),
                a=float(a),
            )

    frontier_mask = delta_grid < 0.01
    return {
        "alpha_grid": alpha_grid,
        "gamma_grid": gamma_grid,
        "delta_grid": delta_grid,
        "frontier_mask": frontier_mask,
        "params": {
            "alpha_range": tuple(alpha_range),
            "gamma_range": tuple(gamma_range),
            "grid_size": int(grid_size),
            "sigma_a": float(sigma_a),
            "sigma_y": float(sigma_y),
            "a": float(a),
        },
    }
