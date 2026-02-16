from pathlib import Path
import csv
import json

from real_data.utils import save_metrics


def test_save_metrics_json(tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    save_metrics(out, {"delta": 0.1, "nested": {"a": 1}})
    loaded = json.loads(out.read_text())
    assert loaded["delta"] == 0.1
    assert loaded["nested"]["a"] == 1


def test_save_metrics_csv(tmp_path: Path) -> None:
    out = tmp_path / "metrics.csv"
    save_metrics(out, {"overall": {"delta_propensity": 0.2}, "delta_nc": 0.1})
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {row["group"] for row in rows}
    assert "overall" in groups
    assert "delta_nc" in groups
