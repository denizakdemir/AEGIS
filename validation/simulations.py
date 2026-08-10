"""Simulation validation harness for AEGIS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import json
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis import AEGISMonitor, deficiency_proxy_propensity, uniform_deficiency_proxy


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class CurvePoint:
    strength: float
    mean: float
    std: float


def overlap_curve(
    strengths: Iterable[float],
    *,
    n: int = 8000,
    repeats: int = 8,
    seed: int = 17,
    metric: str = "tv",
) -> list[CurvePoint]:
    rng = np.random.default_rng(seed)
    points: list[CurvePoint] = []
    for s in strengths:
        values = []
        for _ in range(repeats):
            w = rng.normal(size=(n, 3))
            p = _sigmoid(float(s) * w[:, 0])
            a = rng.binomial(1, p, size=n)
            delta, _ = deficiency_proxy_propensity(w, a, bins=40, metric=metric)
            values.append(float(delta))
        values_arr = np.asarray(values, dtype=float)
        points.append(
            CurvePoint(
                strength=float(s),
                mean=float(values_arr.mean()),
                std=float(values_arr.std(ddof=1)),
            )
        )
    return points


def negative_control_curve(
    strengths: Iterable[float],
    *,
    n: int = 10000,
    repeats: int = 8,
    seed: int = 23,
    metric: str = "tv",
) -> list[CurvePoint]:
    rng = np.random.default_rng(seed)
    points: list[CurvePoint] = []
    for s in strengths:
        values = []
        for _ in range(repeats):
            u = rng.normal(size=n)
            w = rng.normal(size=(n, 2))
            p = _sigmoid(0.8 * w[:, 0] + float(s) * u)
            a = rng.binomial(1, p, size=n)

            mon = AEGISMonitor(window_size=n, bins=50, metric_nc=metric)
            y_nc = 0.9 * u + rng.normal(scale=0.7, size=n)
            mon.update(w, a, y_nc=y_nc)
            values.append(float(mon.estimate()["delta_nc"]))
        values_arr = np.asarray(values, dtype=float)
        points.append(
            CurvePoint(
                strength=float(s),
                mean=float(values_arr.mean()),
                std=float(values_arr.std(ddof=1)),
            )
        )
    return points


def uniform_curve(
    strengths: Iterable[float],
    *,
    n: int = 9000,
    repeats: int = 6,
    seed: int = 31,
    metric: str = "tv",
) -> list[CurvePoint]:
    rng = np.random.default_rng(seed)
    points: list[CurvePoint] = []
    for s in strengths:
        values = []
        for _ in range(repeats):
            w = rng.normal(size=(n, 2))
            logits = np.stack(
                [
                    float(s) * (1.3 * w[:, 0]),
                    float(s) * (1.1 * w[:, 1]),
                    float(s) * (-0.8 * w[:, 0] - 0.5 * w[:, 1]),
                ],
                axis=1,
            )
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            actions = np.array([rng.choice(3, p=p) for p in probs], dtype=int)
            delta_uniform, _ = uniform_deficiency_proxy(w, actions, bins=30, metric=metric)
            values.append(float(delta_uniform))

        values_arr = np.asarray(values, dtype=float)
        points.append(
            CurvePoint(
                strength=float(s),
                mean=float(values_arr.mean()),
                std=float(values_arr.std(ddof=1)),
            )
        )
    return points


def gating_performance(
    *,
    n_windows: int = 50,
    window_size: int = 1000,
    seed: int = 101,
    delta_propensity_max: float = 0.20,
    delta_nc_max: float = 0.20,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    safe_allow = 0
    risky_abstain = 0

    for _ in range(n_windows):
        w_safe = rng.normal(size=(window_size, 3))
        a_safe = rng.binomial(1, 0.5, size=window_size)
        u_safe = rng.normal(size=window_size)
        y_nc_safe = 0.2 * u_safe + rng.normal(size=window_size)

        mon_safe = AEGISMonitor(
            window_size=window_size,
            bins=40,
            delta_propensity_max=delta_propensity_max,
            delta_nc_max=delta_nc_max,
            kappa=1.0,
            missing_nc_status="review",
        )
        mon_safe.update(w_safe, a_safe, y_nc=y_nc_safe)
        safe_allow += int(mon_safe.assess(record=False)["status"] == "allow")

        u_risky = rng.normal(size=window_size)
        w_risky = rng.normal(size=(window_size, 3))
        p_risky = _sigmoid(1.7 * w_risky[:, 0] + 1.5 * u_risky)
        a_risky = rng.binomial(1, p_risky, size=window_size)
        y_nc_risky = 0.9 * u_risky + rng.normal(scale=0.6, size=window_size)

        mon_risky = AEGISMonitor(
            window_size=window_size,
            bins=40,
            delta_propensity_max=delta_propensity_max,
            delta_nc_max=delta_nc_max,
            kappa=1.0,
            missing_nc_status="review",
        )
        mon_risky.update(w_risky, a_risky, y_nc=y_nc_risky)
        risky_abstain += int(mon_risky.assess(record=False)["status"] == "abstain")

    return {
        "safe_allow_rate": safe_allow / n_windows,
        "risky_abstain_rate": risky_abstain / n_windows,
        "n_windows": float(n_windows),
    }


def run_validation(
    out_dir: Path,
    *,
    n_scale: float = 1.0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
    overlap = overlap_curve(strengths, n=max(2000, int(8000 * n_scale)))
    nc = negative_control_curve(strengths, n=max(2500, int(10000 * n_scale)))
    uniform = uniform_curve(strengths, n=max(2500, int(9000 * n_scale)))
    gating = gating_performance(
        n_windows=max(12, int(50 * n_scale)),
        window_size=max(500, int(1000 * n_scale)),
        delta_nc_max=0.20 if n_scale >= 1.0 else 0.22,
    )

    summary = {
        "overlap_curve": [p.__dict__ for p in overlap],
        "negative_control_curve": [p.__dict__ for p in nc],
        "uniform_curve": [p.__dict__ for p in uniform],
        "gating_performance": gating,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (out_dir / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["curve", "strength", "mean", "std"])
        writer.writeheader()
        for curve_name, points in [
            ("overlap", overlap),
            ("negative_control", nc),
            ("uniform", uniform),
        ]:
            for p in points:
                writer.writerow(
                    {
                        "curve": curve_name,
                        "strength": p.strength,
                        "mean": p.mean,
                        "std": p.std,
                    }
                )

    report = [
        "# AEGIS Synthetic Validation Report",
        "",
        "## Gating performance",
        f"- Low-risk allow rate: {gating['safe_allow_rate']:.3f}",
        f"- High-risk abstain rate: {gating['risky_abstain_rate']:.3f}",
        "",
        "## Overlap curve means",
    ]
    report.extend([f"- strength={p.strength:.1f}: {p.mean:.3f} +/- {p.std:.3f}" for p in overlap])
    report.append("")
    report.append("## Negative-control curve means")
    report.extend([f"- strength={p.strength:.1f}: {p.mean:.3f} +/- {p.std:.3f}" for p in nc])
    report.append("")
    report.append("## Uniform proxy curve means")
    report.extend([f"- strength={p.strength:.1f}: {p.mean:.3f} +/- {p.std:.3f}" for p in uniform])
    report.append("")
    report.append(
        "These are proxy diagnostics under synthetic SCM-inspired simulations, not population-identification proofs or universal threshold guarantees."
    )
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    return summary
