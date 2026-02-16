from __future__ import annotations

import numpy as np

from real_data.reporting import build_proxy_table_rows, summarize_case


def test_summarize_case_with_negative_control() -> None:
    rng = np.random.default_rng(11)
    n = 300
    w = rng.normal(size=(n, 3))
    logits = 0.7 * w[:, 0] - 0.4 * w[:, 1]
    p = 1.0 / (1.0 + np.exp(-logits))
    a = rng.binomial(1, p, size=n)
    y_nc = 0.8 * w[:, 0] + rng.normal(scale=0.5, size=n)

    summary = summarize_case(
        w,
        a,
        y_nc=y_nc,
        bootstrap_reps=12,
        random_state=3,
    )

    assert "single_run" in summary
    assert "delta_propensity" in summary["single_run"]
    assert "delta_nc" in summary["single_run"]
    assert summary["negative_control"]["available"] is True
    assert summary["negative_control"]["reason"] is None
    assert set(summary["metric_sensitivity"].keys()) == {"tv", "w1", "mmd"}
    ci_prop = summary["uncertainty"]["delta_propensity"]
    assert ci_prop["ci_low"] <= ci_prop["ci_high"]
    assert ci_prop["estimate"] is not None
    ci_nc = summary["uncertainty"]["delta_nc"]
    assert ci_nc["ci_low"] <= ci_nc["ci_high"]
    assert ci_nc["estimate"] is not None


def test_summarize_case_without_negative_control_records_reason() -> None:
    rng = np.random.default_rng(13)
    n = 220
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, 0.55, size=n)
    reason = "No validated negative-control outcome is available for this dataset."

    summary = summarize_case(
        w,
        a,
        y_nc=None,
        missing_nc_reason=reason,
        bootstrap_reps=8,
        random_state=1,
    )

    assert "delta_propensity" in summary["single_run"]
    assert "delta_nc" not in summary["single_run"]
    assert summary["negative_control"]["available"] is False
    assert summary["negative_control"]["reason"] == reason
    assert "delta_nc" not in summary["uncertainty"]
    assert all("delta_nc" not in m for m in summary["metric_sensitivity"].values())


def test_build_proxy_table_rows_uses_missing_nc_reason() -> None:
    rows = build_proxy_table_rows(
        {
            "lalonde": {
                "single_run": {"delta_propensity": 0.2, "delta_nc": 0.1},
                "uncertainty": {"delta_propensity": {"ci_low": 0.1, "ci_high": 0.3}},
                "negative_control": {"available": True, "reason": None},
            },
            "folktables": {
                "single_run": {"delta_propensity": 0.9},
                "negative_control": {"available": False, "reason": "No NC available."},
            },
        }
    )
    assert rows[0]["delta_nc"] == "0.100"
    assert rows[1]["delta_nc"] == "N/A"
    assert "No NC available." in rows[1]["nc_status"]
