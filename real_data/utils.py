"""Utilities for AEGIS real-data case studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import csv


def save_metrics(output_path: Path, metrics: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in {".json"}:
        output_path.write_text(json.dumps(metrics, indent=2))
        return

    # Flatten dict for CSV if possible.
    if output_path.suffix.lower() in {".csv"}:
        rows = []
        for key, value in metrics.items():
            if isinstance(value, dict):
                row = {"group": key, **value}
            else:
                row = {"group": key, "value": value}
            rows.append(row)

        if not rows:
            output_path.write_text("")
            return

        field_set = set()
        for row in rows:
            field_set.update(row.keys())
        fieldnames = ["group"] + sorted(k for k in field_set if k != "group")

        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    output_path.write_text(json.dumps(metrics, indent=2))
