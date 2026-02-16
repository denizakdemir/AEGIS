"""Lalonde NSW case study.

Loads the Lalonde/NSW job training data from a public Python package when
available, or from a local CSV. Produces AEGIS metrics and a CSV summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .utils import save_metrics
    from .reporting import summarize_case
except ImportError:
    from utils import save_metrics
    from reporting import summarize_case


def _load_from_causaldata() -> pd.DataFrame:
    try:
        from causaldata import nsw_mixtape  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError("causaldata is not installed") from exc

    if hasattr(nsw_mixtape, "load_pandas"):
        loaded = nsw_mixtape.load_pandas()
        if isinstance(loaded, tuple):
            return loaded[0]
        return loaded
    if hasattr(nsw_mixtape, "data"):
        return nsw_mixtape.data

    raise RuntimeError("Unknown causaldata lalonde loader API")


def _load_from_csv(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def load_lalonde(data_path: Optional[Path] = None) -> pd.DataFrame:
    if data_path is not None:
        return _select_training_partition(_load_from_csv(data_path))
    data = _load_from_causaldata()
    if hasattr(data, "data") and isinstance(data.data, pd.DataFrame):
        return _select_training_partition(data.data)
    if isinstance(data, pd.DataFrame):
        return _select_training_partition(data)
    if hasattr(data, "to_pandas"):
        return _select_training_partition(data.to_pandas())
    if isinstance(data, dict) and "train" in data and hasattr(data["train"], "to_pandas"):
        return _select_training_partition(data["train"].to_pandas())
    raise ValueError("Unsupported Lalonde dataset format")


def _select_training_partition(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer NSW treated/control training split when benchmark variants include extras."""
    if "sample_id" in df.columns:
        sid = df["sample_id"].astype(str)
        mask = sid.isin({"nsw_treated", "nsw_control"})
        if int(mask.sum()) >= 100:
            return df.loc[mask].copy()

    if "data_id" in df.columns:
        did = df["data_id"].astype(str)
        mask = did.str.contains("Dehejia", case=False, na=False)
        if int(mask.sum()) >= 100:
            return df.loc[mask].copy()

    return df


def _extract_columns(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Common column conventions in Lalonde datasets.
    treatment_cols = ["treat", "treatment", "A"]
    outcome_cols = ["re78", "earnings", "Y"]
    nc_cols = ["re74", "re75", "pre_earnings", "Y_nc"]

    def first_present(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    a_col = first_present(treatment_cols)
    y_col = first_present(outcome_cols)
    nc_col = first_present(nc_cols)
    if a_col is None or y_col is None or nc_col is None:
        raise ValueError(
            "Expected columns for treatment (treat), outcome (re78), and negative control (re74/re75)."
        )

    drop_cols = {a_col, y_col, nc_col}
    w_cols = [c for c in df.columns if c not in drop_cols]
    w_df = df[w_cols]
    w_df = w_df.select_dtypes(include=[np.number])
    if w_df.empty:
        raise ValueError("No numeric covariates found in the Lalonde dataset")

    w = w_df.to_numpy()
    a = pd.to_numeric(df[a_col], errors="coerce").fillna(0).to_numpy()
    y_nc = pd.to_numeric(df[nc_col], errors="coerce").fillna(0).to_numpy()

    return w, a, y_nc


def run_case(
    data_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    bootstrap_reps: int = 20,
) -> dict:
    df = load_lalonde(data_path)
    w, a, y_nc = _extract_columns(df)

    metrics = summarize_case(
        w,
        a,
        y_nc=y_nc,
        bins=50,
        bootstrap_reps=bootstrap_reps,
        random_state=7,
    )

    if output_path is not None:
        save_metrics(output_path, metrics)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Lalonde NSW AEGIS case study")
    parser.add_argument("--data", type=Path, default=None, help="Path to local Lalonde CSV")
    parser.add_argument("--out", type=Path, default=Path("lalonde_metrics.csv"))
    parser.add_argument("--bootstrap-reps", type=int, default=20, help="Bootstrap repetitions for CI")
    args = parser.parse_args()

    metrics = run_case(data_path=args.data, output_path=args.out, bootstrap_reps=args.bootstrap_reps)
    print(metrics)
