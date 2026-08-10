"""CLI entry-point for synthetic validation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from simulations import run_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AEGIS synthetic validation suite")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation/out"),
        help="Output directory for synthetic validation artifacts",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a smaller simulation budget for quick checks",
    )
    args = parser.parse_args()

    n_scale = 0.35 if args.fast else 1.0
    summary = run_validation(args.out_dir, n_scale=n_scale)

    gating = summary["gating_performance"]
    print(f"[OK] Wrote synthetic validation outputs to {args.out_dir}")
    print(
        f"[OK] low_risk_allow_rate={gating['safe_allow_rate']:.3f}, "
        f"high_risk_abstain_rate={gating['risky_abstain_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
