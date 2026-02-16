"""Detailed reporting helpers for AEGIS real-data case studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import csv
import numpy as np


DEFAULT_METRICS = ("tv", "w1", "mmd")
DEFAULT_MISSING_NC_REASON = "No validated negative-control outcome is available for this dataset."


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("Expected 1D or 2D features.")
    return x


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _compute_metrics(
    w: np.ndarray,
    a: np.ndarray,
    *,
    y_nc: Optional[np.ndarray],
    bins: int,
    metric_propensity: str,
    metric_nc: str,
    random_state: int,
) -> Dict[str, float]:
    from aegis import AEGISMonitor

    monitor = AEGISMonitor(
        window_size=int(a.shape[0]),
        bins=bins,
        metric_propensity=metric_propensity,
        metric_nc=metric_nc,
        random_state=random_state,
    )
    monitor.update(w, a, y_nc=y_nc)
    return monitor.estimate()


def _distribution_summary(
    values: list[float],
    *,
    estimate: Optional[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "estimate": float(estimate) if estimate is not None else None,
        "n_boot": 0,
        "mean": None,
        "std": None,
        "ci_low": None,
        "ci_high": None,
    }
    if not values:
        return out

    arr = np.asarray(values, dtype=float)
    out["n_boot"] = int(arr.size)
    out["mean"] = float(np.mean(arr))
    out["std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    out["ci_low"] = float(np.percentile(arr, 2.5))
    out["ci_high"] = float(np.percentile(arr, 97.5))
    return out


def summarize_case(
    w: np.ndarray,
    a: np.ndarray,
    *,
    y_nc: Optional[np.ndarray] = None,
    bins: int = 50,
    bootstrap_reps: int = 20,
    bootstrap_sample_size: Optional[int] = None,
    metrics: Iterable[str] = DEFAULT_METRICS,
    random_state: int = 7,
    missing_nc_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Return detailed single-run, uncertainty, and metric-sensitivity summaries."""
    w = _as_2d(w)
    a = _as_1d(a)
    if w.shape[0] != a.shape[0]:
        raise ValueError("Feature and action lengths do not match.")
    y_nc_vec = None if y_nc is None else _as_1d(y_nc)
    if y_nc_vec is not None and y_nc_vec.shape[0] != a.shape[0]:
        raise ValueError("Negative-control length does not match actions.")

    single_run = _compute_metrics(
        w,
        a,
        y_nc=y_nc_vec,
        bins=bins,
        metric_propensity="tv",
        metric_nc="tv",
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    sample_size = int(bootstrap_sample_size or a.shape[0])
    sample_size = max(2, min(sample_size, a.shape[0]))
    max_draws = max(bootstrap_reps * 6, 30)

    prop_boot: list[float] = []
    nc_boot: list[float] = []
    draws = 0
    while draws < max_draws and len(prop_boot) < bootstrap_reps:
        draws += 1
        idx = rng.choice(a.shape[0], size=sample_size, replace=True)
        a_b = a[idx]
        # Skip degenerate resamples with one action arm.
        if np.unique(a_b).size < 2:
            continue
        y_nc_b = None if y_nc_vec is None else y_nc_vec[idx]
        metrics_b = _compute_metrics(
            w[idx],
            a_b,
            y_nc=y_nc_b,
            bins=bins,
            metric_propensity="tv",
            metric_nc="tv",
            random_state=random_state + draws,
        )
        prop_boot.append(float(metrics_b["delta_propensity"]))
        if "delta_nc" in metrics_b:
            nc_boot.append(float(metrics_b["delta_nc"]))

    uncertainty: Dict[str, Dict[str, Optional[float]]] = {
        "delta_propensity": _distribution_summary(
            prop_boot,
            estimate=float(single_run["delta_propensity"]),
        )
    }
    if "delta_nc" in single_run:
        uncertainty["delta_nc"] = _distribution_summary(
            nc_boot,
            estimate=float(single_run["delta_nc"]),
        )

    metric_sensitivity: Dict[str, Dict[str, float]] = {}
    for metric_name in metrics:
        metrics_now = _compute_metrics(
            w,
            a,
            y_nc=y_nc_vec,
            bins=bins,
            metric_propensity=str(metric_name),
            metric_nc=str(metric_name),
            random_state=random_state,
        )
        metric_sensitivity[str(metric_name)] = metrics_now

    if y_nc_vec is None:
        nc_reason = missing_nc_reason or DEFAULT_MISSING_NC_REASON
    else:
        nc_reason = None

    result: Dict[str, Any] = {
        **single_run,
        "single_run": single_run,
        "uncertainty": uncertainty,
        "metric_sensitivity": metric_sensitivity,
        "negative_control": {
            "available": y_nc_vec is not None,
            "reason": nc_reason,
        },
    }
    return result


def build_proxy_table_rows(results: Dict[str, Any]) -> list[Dict[str, str]]:
    """Create manuscript-friendly rows from run_all case summaries."""
    rows: list[Dict[str, str]] = []
    labels = {
        "lalonde": "Lalonde NSW (training)",
        "folktables": "Folktables ACS Income (overall)",
        "amazon": "Amazon Reviews (overall)",
    }
    for key in ("lalonde", "folktables", "amazon"):
        summary = results.get(key)
        if not summary:
            continue

        if isinstance(summary, dict) and "overall" in summary and isinstance(summary["overall"], dict):
            overall = summary["overall"]
        else:
            overall = summary

        single_run = overall.get("single_run", overall)
        dp = single_run.get("delta_propensity")
        dn = single_run.get("delta_nc")
        nc_meta = overall.get("negative_control", {})
        nc_available = bool(nc_meta.get("available", dn is not None))
        nc_reason = nc_meta.get("reason")
        if (not nc_available) and not nc_reason:
            if key == "folktables":
                nc_reason = "No validated negative-control outcome is used in this ACS policy proxy setup."
            elif key == "amazon":
                nc_reason = (
                    "No validated negative-control outcome is available in the sentiment benchmark setup."
                )
            else:
                nc_reason = DEFAULT_MISSING_NC_REASON

        unc = overall.get("uncertainty", {}).get("delta_propensity", {})
        ci_low = unc.get("ci_low")
        ci_high = unc.get("ci_high")
        ci_text = "-"
        if ci_low is not None and ci_high is not None:
            ci_text = f"[{float(ci_low):.3f}, {float(ci_high):.3f}]"

        dn_text = f"{float(dn):.3f}" if dn is not None else "N/A"
        nc_note = "available" if nc_available else "missing"
        if (not nc_available) and nc_reason:
            nc_note = f"missing: {nc_reason}"

        row = {
            "dataset": labels.get(key, key),
            "delta_propensity": f"{float(dp):.3f}" if dp is not None else "N/A",
            "delta_propensity_ci95": ci_text,
            "delta_nc": dn_text,
            "nc_status": nc_note,
        }
        rows.append(row)
    return rows


def write_proxy_table(rows: list[Dict[str, str]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "delta_propensity", "delta_propensity_ci95", "delta_nc", "nc_status"]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# AEGIS Real-Data Proxy Table",
        "",
        "| Dataset | delta_propensity | 95% bootstrap CI | delta_nc | NC status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['delta_propensity']} | {row['delta_propensity_ci95']} "
            f"| {row['delta_nc']} | {row['nc_status']} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
