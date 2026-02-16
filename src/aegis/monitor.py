"""AEGIS monitoring utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .bounds import negative_control_sensitivity_bound
from .distances import distance_1d, validate_metric


def _validate_binary(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if values.dtype == bool:
        return values.astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    unique = np.unique(values)
    if unique.size > 2 or not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"{name} must be binary (0/1). Got values {unique}.")
    return values.astype(float)


def _as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Expected 1D or 2D array for features.")
    return array


def _validate_threshold(value: Optional[float], *, name: str) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _add_intercept(x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1)), x])


def fit_logistic_newton(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> np.ndarray:
    """Fit binary logistic regression with a Newton method."""
    x = _add_intercept(_as_2d(x))
    y = _validate_binary(y, name="y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Feature and label lengths do not match.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Features must be finite.")

    coef = np.zeros(x.shape[1])
    for _ in range(max_iter):
        logits = x @ coef
        p = _sigmoid(logits)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)

        grad = x.T @ (p - y)
        grad[1:] += l2 * coef[1:]
        w = p * (1.0 - p)
        hess = x.T @ (x * w[:, None])
        if x.shape[1] > 1:
            hess[1:, 1:] += l2 * np.eye(x.shape[1] - 1)

        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        coef -= step

        if np.linalg.norm(step, ord=2) < tol:
            break

    return coef


def predict_propensity(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Predict propensity scores from binary logistic regression coefficients."""
    x = _add_intercept(_as_2d(x))
    return _sigmoid(x @ coef)


def deficiency_proxy_propensity(
    x: np.ndarray,
    a: np.ndarray,
    bins: int = 50,
    l2: float = 1e-3,
    max_iter: int = 25,
    tol: float = 1e-6,
    metric: str = "tv",
    mmd_gamma: Optional[float] = None,
    max_mmd_samples: int = 512,
    random_state: int = 7,
) -> Tuple[float, np.ndarray]:
    """Estimate propensity-overlap deficiency proxy for binary actions."""
    metric = validate_metric(metric)
    a = _validate_binary(a, name="a")
    coef = fit_logistic_newton(x, a, l2=l2, max_iter=max_iter, tol=tol)
    e = predict_propensity(x, coef)
    delta = distance_1d(
        e[a == 1],
        e[a == 0],
        metric=metric,
        bins=bins,
        value_range=(0.0, 1.0),
        mmd_gamma=mmd_gamma,
        max_mmd_samples=max_mmd_samples,
        random_state=random_state,
    )
    return float(delta), e


def uniform_deficiency_proxy(
    x: np.ndarray,
    a: np.ndarray,
    bins: int = 50,
    l2: float = 1e-3,
    max_iter: int = 25,
    tol: float = 1e-6,
    metric: str = "tv",
    mmd_gamma: Optional[float] = None,
    max_mmd_samples: int = 512,
    random_state: int = 7,
) -> Tuple[float, Dict[str, float]]:
    """One-vs-rest proxy for uniform deficiency across discrete actions."""
    validate_metric(metric)
    x = _as_2d(x)
    a = np.asarray(a).reshape(-1)
    if x.shape[0] != a.shape[0]:
        raise ValueError("Feature and action lengths do not match.")
    if a.size == 0:
        raise ValueError("a must be non-empty.")

    unique_actions = np.unique(a)
    if unique_actions.size < 2:
        raise ValueError("Need at least two unique actions to compute uniform deficiency.")

    per_action: Dict[str, float] = {}
    for action_value in unique_actions:
        a_binary = (a == action_value).astype(float)
        if np.sum(a_binary) == 0 or np.sum(1.0 - a_binary) == 0:
            continue
        delta, _ = deficiency_proxy_propensity(
            x,
            a_binary,
            bins=bins,
            l2=l2,
            max_iter=max_iter,
            tol=tol,
            metric=metric,
            mmd_gamma=mmd_gamma,
            max_mmd_samples=max_mmd_samples,
            random_state=random_state,
        )
        per_action[str(action_value)] = float(delta)

    if not per_action:
        return float("nan"), {}

    return float(max(per_action.values())), per_action


def negative_control_distance(
    y_nc: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    bins: int = 50,
    max_weight: Optional[float] = None,
    metric: str = "tv",
    mmd_gamma: Optional[float] = None,
    max_mmd_samples: int = 512,
    random_state: int = 7,
) -> float:
    """Estimate IPTW discrepancy on a negative-control outcome."""
    metric = validate_metric(metric)
    y_nc = np.asarray(y_nc).reshape(-1)
    a = _validate_binary(a, name="a")
    e = np.asarray(e).reshape(-1)

    if y_nc.shape[0] != a.shape[0] or a.shape[0] != e.shape[0]:
        raise ValueError("Negative control, action, and propensity lengths must match.")
    if not np.all(np.isfinite(y_nc)):
        raise ValueError("Negative control values must be finite.")

    e = np.clip(e, 1e-6, 1.0 - 1e-6)
    weights = a / e + (1.0 - a) / (1.0 - e)
    if max_weight is not None:
        if max_weight <= 0:
            raise ValueError("max_weight must be positive when provided.")
        weights = np.minimum(weights, max_weight)

    values_t = y_nc[a == 1]
    values_c = y_nc[a == 0]
    weights_t = weights[a == 1]
    weights_c = weights[a == 0]

    y_min = float(np.min(y_nc))
    y_max = float(np.max(y_nc))
    if y_min == y_max:
        return 0.0

    return float(
        distance_1d(
            values_t,
            values_c,
            metric=metric,
            bins=bins,
            value_range=(y_min, y_max),
            weights_a=weights_t,
            weights_b=weights_c,
            mmd_gamma=mmd_gamma,
            max_mmd_samples=max_mmd_samples,
            random_state=random_state,
        )
    )


def negative_control_tv(
    y_nc: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    bins: int = 50,
    max_weight: Optional[float] = None,
) -> float:
    """Backward-compatible TV-only negative-control diagnostic."""
    return negative_control_distance(
        y_nc=y_nc,
        a=a,
        e=e,
        bins=bins,
        max_weight=max_weight,
        metric="tv",
    )


@dataclass
class AEGISMonitor:
    """Streaming monitor with optional threshold-based gating."""

    window_size: int = 5000
    bins: int = 50
    l2: float = 1e-3
    max_iter: int = 25
    tol: float = 1e-6
    max_weight: Optional[float] = None
    metric_propensity: str = "tv"
    metric_nc: str = "tv"
    mmd_gamma: Optional[float] = None
    max_mmd_samples: int = 512
    random_state: int = 7
    delta_propensity_max: Optional[float] = None
    delta_nc_max: Optional[float] = None
    kappa: float = 1.0
    missing_nc_status: str = "review"

    _w_batches: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _a_batches: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _nc_batches: Deque[Optional[np.ndarray]] = field(default_factory=deque, init=False)
    _count: int = field(default=0, init=False)
    _decision_log: list[dict] = field(default_factory=list, init=False)
    _decision_seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.bins <= 1:
            raise ValueError("bins must be >= 2.")
        if self.max_mmd_samples <= 0:
            raise ValueError("max_mmd_samples must be positive.")
        self.metric_propensity = validate_metric(self.metric_propensity)
        self.metric_nc = validate_metric(self.metric_nc)
        self.delta_propensity_max = _validate_threshold(
            self.delta_propensity_max,
            name="delta_propensity_max",
        )
        self.delta_nc_max = _validate_threshold(
            self.delta_nc_max,
            name="delta_nc_max",
        )
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.missing_nc_status not in {"allow", "review", "abstain"}:
            raise ValueError("missing_nc_status must be one of {'allow','review','abstain'}.")

    def set_thresholds(
        self,
        *,
        delta_propensity_max: Optional[float] = None,
        delta_nc_max: Optional[float] = None,
        kappa: Optional[float] = None,
        missing_nc_status: Optional[str] = None,
    ) -> None:
        """Set or update gating thresholds."""
        self.delta_propensity_max = _validate_threshold(
            delta_propensity_max,
            name="delta_propensity_max",
        )
        self.delta_nc_max = _validate_threshold(delta_nc_max, name="delta_nc_max")
        if kappa is not None:
            if kappa <= 0:
                raise ValueError("kappa must be positive.")
            self.kappa = float(kappa)
        if missing_nc_status is not None:
            if missing_nc_status not in {"allow", "review", "abstain"}:
                raise ValueError("missing_nc_status must be one of {'allow','review','abstain'}.")
            self.missing_nc_status = missing_nc_status

    def update(
        self, x: np.ndarray, a: np.ndarray, y_nc: Optional[np.ndarray] = None
    ) -> None:
        x = _as_2d(x)
        a = _validate_binary(a, name="a")
        if x.shape[0] != a.shape[0]:
            raise ValueError("Feature and action lengths do not match.")
        if y_nc is not None:
            y_nc = np.asarray(y_nc).reshape(-1)
            if y_nc.shape[0] != a.shape[0]:
                raise ValueError("Negative control length does not match actions.")
            if not np.all(np.isfinite(y_nc)):
                raise ValueError("Negative control values must be finite.")

        self._w_batches.append(x)
        self._a_batches.append(a)
        self._nc_batches.append(y_nc)
        self._count += a.shape[0]

        self._trim_to_window()

    def _trim_to_window(self) -> None:
        while self._count > self.window_size and self._w_batches:
            excess = self._count - self.window_size
            w0 = self._w_batches[0]
            a0 = self._a_batches[0]
            nc0 = self._nc_batches[0]
            n0 = a0.shape[0]

            if excess >= n0:
                self._w_batches.popleft()
                self._a_batches.popleft()
                self._nc_batches.popleft()
                self._count -= n0
                continue

            self._w_batches[0] = w0[excess:]
            self._a_batches[0] = a0[excess:]
            if nc0 is not None:
                self._nc_batches[0] = nc0[excess:]
            self._count -= excess

    def _stack(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not self._w_batches:
            raise ValueError("No data available in the monitor window.")

        w = np.vstack(list(self._w_batches))
        a = np.concatenate(list(self._a_batches))

        if any(batch is None for batch in self._nc_batches):
            return w, a, None

        nc = np.concatenate([batch for batch in self._nc_batches if batch is not None])
        return w, a, nc

    def estimate(self) -> Dict[str, float]:
        """Estimate overlap and optional negative-control diagnostics."""
        w, a, y_nc = self._stack()
        delta_propensity, e = deficiency_proxy_propensity(
            w,
            a,
            bins=self.bins,
            l2=self.l2,
            max_iter=self.max_iter,
            tol=self.tol,
            metric=self.metric_propensity,
            mmd_gamma=self.mmd_gamma,
            max_mmd_samples=self.max_mmd_samples,
            random_state=self.random_state,
        )

        results = {"delta_propensity": float(delta_propensity)}
        if y_nc is not None:
            delta_nc = float(
                negative_control_distance(
                    y_nc,
                    a,
                    e,
                    bins=self.bins,
                    max_weight=self.max_weight,
                    metric=self.metric_nc,
                    mmd_gamma=self.mmd_gamma,
                    max_mmd_samples=self.max_mmd_samples,
                    random_state=self.random_state,
                )
            )
            results["delta_nc"] = delta_nc
            results["delta_nc_upper_bound"] = float(
                negative_control_sensitivity_bound(delta_nc, kappa=self.kappa)
            )
        return results

    def assess(self, *, record: bool = True) -> Dict[str, object]:
        """Return threshold-based status suitable for gating/abstention decisions."""
        metrics = self.estimate()
        status = "allow"
        reasons = []

        if self.delta_propensity_max is not None:
            if metrics["delta_propensity"] > self.delta_propensity_max:
                status = "abstain"
                reasons.append(f"delta_propensity>{self.delta_propensity_max:.3f}")

        if self.delta_nc_max is not None:
            if "delta_nc" not in metrics:
                if status != "abstain":
                    status = self.missing_nc_status
                reasons.append("delta_nc_missing")
            elif metrics["delta_nc"] > self.delta_nc_max:
                status = "abstain"
                reasons.append(f"delta_nc>{self.delta_nc_max:.3f}")

        decision = {
            "status": status,
            "reasons": reasons,
            "metrics": metrics,
            "window_count": self._count,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if record:
            self._decision_seq += 1
            decision["id"] = self._decision_seq
            self._decision_log.append(decision)
        return decision

    def decision_log(self) -> list[dict]:
        """Return a copy of the decision log."""
        return list(self._decision_log)

    def clear_decision_log(self) -> None:
        """Clear decision history."""
        self._decision_log.clear()
        self._decision_seq = 0

