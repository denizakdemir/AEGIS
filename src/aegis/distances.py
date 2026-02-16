"""Distance utilities for AEGIS proxy diagnostics."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


VALID_METRICS = {"tv", "w1", "mmd"}


def validate_metric(metric: str) -> str:
    metric = str(metric).lower().strip()
    if metric not in VALID_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Use one of {sorted(VALID_METRICS)}.")
    return metric


def hist_tv(
    values_a: np.ndarray,
    values_b: np.ndarray,
    bins: int = 50,
    value_range: Tuple[float, float] = (0.0, 1.0),
    weights_a: Optional[np.ndarray] = None,
    weights_b: Optional[np.ndarray] = None,
) -> float:
    """Histogram-based TV distance in [0, 1]."""
    if values_a.size == 0 or values_b.size == 0:
        return float("nan")

    hist_a, _ = np.histogram(
        values_a, bins=bins, range=value_range, weights=weights_a, density=False
    )
    hist_b, _ = np.histogram(
        values_b, bins=bins, range=value_range, weights=weights_b, density=False
    )

    total_a = np.sum(hist_a)
    total_b = np.sum(hist_b)
    if total_a == 0 or total_b == 0:
        return float("nan")

    p_a = hist_a / total_a
    p_b = hist_b / total_b
    return float(0.5 * np.sum(np.abs(p_a - p_b)))


def _normalize_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape[0] != n:
        raise ValueError("Weights length does not match values length.")
    weights = np.clip(weights, 0.0, np.inf)
    total = float(np.sum(weights))
    if total <= 0:
        return np.full(n, 1.0 / n, dtype=float)
    return weights / total


def wasserstein_1_normalized(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    weights_a: Optional[np.ndarray] = None,
    weights_b: Optional[np.ndarray] = None,
    norm_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Weighted 1D Wasserstein distance normalized to [0, 1] by value range."""
    values_a = np.asarray(values_a, dtype=float).reshape(-1)
    values_b = np.asarray(values_b, dtype=float).reshape(-1)
    if values_a.size == 0 or values_b.size == 0:
        return float("nan")

    wa = _normalize_weights(weights_a, values_a.size)
    wb = _normalize_weights(weights_b, values_b.size)

    order_a = np.argsort(values_a)
    order_b = np.argsort(values_b)
    xa = values_a[order_a]
    xb = values_b[order_b]
    wa = wa[order_a]
    wb = wb[order_b]

    cwa = np.cumsum(wa)
    cwb = np.cumsum(wb)

    grid = np.sort(np.unique(np.concatenate([xa, xb])))
    if grid.size <= 1:
        return 0.0

    idx_a = np.searchsorted(xa, grid, side="right") - 1
    idx_b = np.searchsorted(xb, grid, side="right") - 1

    fa = np.where(idx_a >= 0, cwa[np.clip(idx_a, 0, cwa.size - 1)], 0.0)
    fb = np.where(idx_b >= 0, cwb[np.clip(idx_b, 0, cwb.size - 1)], 0.0)

    widths = np.diff(grid)
    if widths.size == 0:
        return 0.0

    w1 = float(np.sum(np.abs(fa[:-1] - fb[:-1]) * widths))

    if norm_range is None:
        lo = float(min(np.min(values_a), np.min(values_b)))
        hi = float(max(np.max(values_a), np.max(values_b)))
    else:
        lo, hi = float(norm_range[0]), float(norm_range[1])
    denom = max(hi - lo, 1e-12)
    return float(np.clip(w1 / denom, 0.0, 1.0))


def _sample_for_mmd(
    values: np.ndarray,
    weights: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = values.shape[0]
    if n <= max_samples:
        return values, weights
    idx = rng.choice(n, size=max_samples, replace=False, p=weights)
    sampled_values = values[idx]
    sampled_weights = weights[idx]
    sampled_weights = sampled_weights / np.sum(sampled_weights)
    return sampled_values, sampled_weights


def mmd_rbf_normalized(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    weights_a: Optional[np.ndarray] = None,
    weights_b: Optional[np.ndarray] = None,
    gamma: Optional[float] = None,
    max_samples: int = 512,
    random_state: int = 7,
) -> float:
    """Weighted RBF-MMD normalized to [0, 1] by dividing by sqrt(2)."""
    values_a = np.asarray(values_a, dtype=float).reshape(-1, 1)
    values_b = np.asarray(values_b, dtype=float).reshape(-1, 1)
    if values_a.size == 0 or values_b.size == 0:
        return float("nan")

    wa = _normalize_weights(weights_a, values_a.shape[0])
    wb = _normalize_weights(weights_b, values_b.shape[0])

    rng = np.random.default_rng(random_state)
    values_a, wa = _sample_for_mmd(values_a, wa, max_samples=max_samples, rng=rng)
    values_b, wb = _sample_for_mmd(values_b, wb, max_samples=max_samples, rng=rng)

    if gamma is None:
        pooled = np.concatenate([values_a[:, 0], values_b[:, 0]])
        if pooled.size <= 1:
            return 0.0
        q75, q25 = np.percentile(pooled, [75, 25])
        scale = float(max(q75 - q25, np.std(pooled), 1e-6))
        gamma = 1.0 / (2.0 * scale * scale)
    else:
        gamma = float(gamma)
        if gamma <= 0:
            raise ValueError("gamma must be positive when provided.")

    dx = values_a - values_a.T
    dy = values_b - values_b.T
    dxy = values_a - values_b.T

    kxx = np.exp(-gamma * (dx * dx))
    kyy = np.exp(-gamma * (dy * dy))
    kxy = np.exp(-gamma * (dxy * dxy))

    term_xx = float(wa @ kxx @ wa)
    term_yy = float(wb @ kyy @ wb)
    term_xy = float(wa @ kxy @ wb)
    mmd2 = max(term_xx + term_yy - 2.0 * term_xy, 0.0)
    mmd = np.sqrt(mmd2)
    return float(np.clip(mmd / np.sqrt(2.0), 0.0, 1.0))


def distance_1d(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    metric: str = "tv",
    bins: int = 50,
    value_range: Optional[Tuple[float, float]] = None,
    weights_a: Optional[np.ndarray] = None,
    weights_b: Optional[np.ndarray] = None,
    mmd_gamma: Optional[float] = None,
    max_mmd_samples: int = 512,
    random_state: int = 7,
) -> float:
    """Unified distance interface for 1D sample diagnostics."""
    metric = validate_metric(metric)
    values_a = np.asarray(values_a, dtype=float).reshape(-1)
    values_b = np.asarray(values_b, dtype=float).reshape(-1)

    if values_a.size == 0 or values_b.size == 0:
        return float("nan")

    if metric == "tv":
        if value_range is None:
            lo = float(min(np.min(values_a), np.min(values_b)))
            hi = float(max(np.max(values_a), np.max(values_b)))
            if hi == lo:
                return 0.0
            value_range = (lo, hi)
        return hist_tv(
            values_a,
            values_b,
            bins=bins,
            value_range=value_range,
            weights_a=weights_a,
            weights_b=weights_b,
        )

    if metric == "w1":
        return wasserstein_1_normalized(
            values_a,
            values_b,
            weights_a=weights_a,
            weights_b=weights_b,
            norm_range=value_range,
        )

    return mmd_rbf_normalized(
        values_a,
        values_b,
        weights_a=weights_a,
        weights_b=weights_b,
        gamma=mmd_gamma,
        max_samples=max_mmd_samples,
        random_state=random_state,
    )

