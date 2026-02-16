"""Run all AEGIS real-data case studies (best-effort).

This helper is designed for manuscript reproducibility. It will:
- run each case when its optional dependency stack is available, and
- write outputs to the provided `out/` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .reporting import build_proxy_table_rows, write_proxy_table
except ImportError:
    from reporting import build_proxy_table_rows, write_proxy_table


def _try(name: str, fn, out_path: Path, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        metrics = fn(output_path=out_path, **kwargs)
    except ImportError as exc:
        print(f"[SKIP] {name}: {exc}")
        cached = _load_cached_metrics(out_path)
        if cached is not None:
            print(f"[CACHE] {name}: loaded existing {out_path}")
        return cached
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        raise
    print(f"[OK] {name}: wrote {out_path}")
    return metrics


def _load_cached_metrics(out_path: Path) -> Optional[Dict[str, Any]]:
    if not out_path.exists():
        return None

    if out_path.suffix.lower() == ".json":
        return json.loads(out_path.read_text(encoding="utf-8"))

    if out_path.suffix.lower() == ".csv":
        with out_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return None

        if "group" in rows[0] and "value" in rows[0]:
            parsed: Dict[str, Any] = {}
            for row in rows:
                key = row.get("group")
                if not key:
                    continue
                value = row.get("value")
                try:
                    parsed[key] = float(value) if value is not None else None
                except (TypeError, ValueError):
                    parsed[key] = value
            return parsed

        return {"rows": rows}

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all AEGIS real-data case studies")
    parser.add_argument("--out-dir", type=Path, default=Path("out"), help="Output directory")
    parser.add_argument("--amazon-dataset", type=str, default=None, help="HF dataset id for Amazon case")
    parser.add_argument("--amazon-sample-size", type=int, default=50000, help="HF rows to load (Amazon)")
    parser.add_argument("--amazon-max-rows", type=int, default=5000, help="Rows used for AEGIS eval (Amazon)")
    parser.add_argument("--amazon-svd-dim", type=int, default=0, help="SVD dimension (0 uses score) (Amazon)")
    parser.add_argument("--folktables-states", nargs="+", default=["CA", "NY", "TX"], help="ACS state codes")
    parser.add_argument(
        "--lalonde-data",
        type=Path,
        default=None,
        help="Optional local Lalonde/NSW CSV path (used when causaldata is unavailable).",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=20,
        help="Bootstrap repetitions for real-data uncertainty summaries",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    try:
        from .lalonde_case import run_case as run_lalonde
    except ImportError:
        from lalonde_case import run_case as run_lalonde  # type: ignore

    results["lalonde"] = _try(
        "Lalonde NSW",
        run_lalonde,
        out_dir / "lalonde_metrics.csv",
        data_path=args.lalonde_data,
        bootstrap_reps=args.bootstrap_reps,
    )

    try:
        from .folktables_case import run_case as run_folktables
    except ImportError:
        from folktables_case import run_case as run_folktables  # type: ignore

    results["folktables"] = _try(
        "Folktables ACS Income",
        run_folktables,
        out_dir / "folktables_metrics.json",
        states=args.folktables_states,
        bootstrap_reps=args.bootstrap_reps,
    )

    try:
        from .amazon_reviews_case import run_case as run_amazon
    except ImportError:
        from amazon_reviews_case import run_case as run_amazon  # type: ignore

    results["amazon"] = _try(
        "Amazon Reviews",
        run_amazon,
        out_dir / "amazon_metrics.json",
        dataset_id=args.amazon_dataset,
        sample_size=args.amazon_sample_size,
        max_rows=args.amazon_max_rows,
        svd_dim=args.amazon_svd_dim,
        bootstrap_reps=args.bootstrap_reps,
    )

    (out_dir / "real_data_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    rows = build_proxy_table_rows(results)
    write_proxy_table(
        rows,
        csv_path=out_dir / "real_data_proxy_table.csv",
        md_path=out_dir / "real_data_proxy_table.md",
    )
    print(f"[OK] Proxy table artifacts: {out_dir / 'real_data_proxy_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
