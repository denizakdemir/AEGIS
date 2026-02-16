"""Folktables ACS Income case study.

Loads ACS data with folktables, trains a baseline classifier, defines a policy
action, and computes AEGIS metrics across regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis import AEGISMonitor

try:
    from .utils import save_metrics
    from .reporting import summarize_case
except ImportError:
    from utils import save_metrics
    from reporting import summarize_case


def load_folktables(states: Optional[list] = None) -> Dict[str, np.ndarray]:
    try:
        from folktables import ACSDataSource, ACSIncome  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError("folktables is not installed") from exc

    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    states = states or ["CA", "NY", "TX"]

    frames = []
    for state in states:
        acs_data = data_source.get_data(states=[state], download=True)
        features, labels, _ = ACSIncome.df_to_numpy(acs_data)
        frame = pd.DataFrame(features, columns=ACSIncome.features)
        frame["label"] = labels
        frame["state"] = state
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    return {
        "W": df[ACSIncome.features].to_numpy(),
        "Y": df["label"].to_numpy(),
        "state": df["state"].to_numpy(),
    }


def simulate_policy(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (scores >= threshold).astype(int)


def run_case(
    states: Optional[list] = None,
    output_path: Optional[Path] = None,
    bootstrap_reps: int = 20,
) -> Dict[str, float]:
    data = load_folktables(states=states)
    w = data["W"]
    y = data["Y"]
    state = data["state"]

    x_train, x_test, y_train, y_test, state_train, state_test = train_test_split(
        w, y, state, test_size=0.3, random_state=42, stratify=state
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, n_jobs=1)),
        ]
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    a = simulate_policy(scores)

    overall = summarize_case(
        x_test,
        a,
        y_nc=None,
        bins=50,
        bootstrap_reps=bootstrap_reps,
        random_state=7,
        missing_nc_reason="No validated negative-control outcome is used in this ACS policy proxy setup.",
    )

    results = {"overall": overall}
    for st in np.unique(state_test):
        mask = state_test == st
        if mask.sum() < 50:
            continue
        monitor = AEGISMonitor(window_size=min(5000, mask.sum()), bins=50)
        monitor.update(x_test[mask], a[mask])
        state_metrics = monitor.estimate()
        state_metrics["negative_control"] = {
            "available": False,
            "reason": "No validated negative-control outcome is used in this ACS policy proxy setup.",
        }
        results[st] = state_metrics

    if output_path is not None:
        save_metrics(output_path, results)

    return overall


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Folktables ACS AEGIS case study")
    parser.add_argument("--states", nargs="+", default=None, help="List of ACS state codes")
    parser.add_argument("--out", type=Path, default=Path("folktables_metrics.json"))
    parser.add_argument("--bootstrap-reps", type=int, default=20, help="Bootstrap repetitions for CI")
    args = parser.parse_args()

    metrics = run_case(states=args.states, output_path=args.out, bootstrap_reps=args.bootstrap_reps)
    print(metrics)
