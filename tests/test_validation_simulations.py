from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from simulations import gating_performance, negative_control_curve, overlap_curve, uniform_curve


def _is_non_decreasing(values: list[float], tol: float = 1e-3) -> bool:
    diffs = np.diff(np.asarray(values, dtype=float))
    return bool(np.all(diffs >= -tol))


def test_overlap_curve_monotone() -> None:
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
    points = overlap_curve(strengths, n=2500, repeats=4, seed=202)
    means = [p.mean for p in points]
    assert _is_non_decreasing(means)
    assert means[-1] > means[0]


def test_negative_control_curve_monotone() -> None:
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
    points = negative_control_curve(strengths, n=2500, repeats=4, seed=303)
    means = [p.mean for p in points]
    assert _is_non_decreasing(means)
    assert means[-1] > means[0]


def test_uniform_curve_monotone() -> None:
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
    points = uniform_curve(strengths, n=3000, repeats=3, seed=404)
    means = [p.mean for p in points]
    assert _is_non_decreasing(means)
    assert means[-1] > means[0]


def test_curves_support_alternative_metrics() -> None:
    strengths = [0.0, 1.0]
    for metric in ("tv", "w1", "mmd"):
        points_o = overlap_curve(strengths, n=1200, repeats=2, seed=606, metric=metric)
        points_n = negative_control_curve(strengths, n=1200, repeats=2, seed=607, metric=metric)
        points_u = uniform_curve(strengths, n=1500, repeats=2, seed=608, metric=metric)
        for pts in (points_o, points_n, points_u):
            for p in pts:
                assert 0.0 <= p.mean <= 1.0


def test_gating_performance_separates_safe_and_risky() -> None:
    perf = gating_performance(n_windows=16, window_size=500, seed=505)
    assert perf["safe_allow_rate"] >= 0.6
    assert perf["risky_abstain_rate"] >= 0.8
