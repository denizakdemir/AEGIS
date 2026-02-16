from pathlib import Path

import numpy as np
import pandas as pd

from real_data import amazon_reviews_case, folktables_case, lalonde_case


def test_lalonde_case_runs_with_local_csv(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    n = 500
    df = pd.DataFrame(
        {
            "treat": rng.binomial(1, 0.5, size=n),
            "re78": rng.normal(loc=1000, scale=300, size=n),
            "re74": rng.normal(loc=900, scale=280, size=n),
            "age": rng.integers(18, 60, size=n),
            "education": rng.integers(8, 20, size=n),
            "re75": rng.normal(loc=950, scale=290, size=n),
        }
    )
    csv_path = tmp_path / "lalonde.csv"
    out_path = tmp_path / "lalonde_metrics.csv"
    df.to_csv(csv_path, index=False)

    metrics = lalonde_case.run_case(data_path=csv_path, output_path=out_path)
    assert "delta_propensity" in metrics
    assert "delta_nc" in metrics
    assert "uncertainty" in metrics
    assert metrics["negative_control"]["available"] is True
    assert out_path.exists()


def test_folktables_simulate_policy_binary() -> None:
    scores = np.array([0.1, 0.4, 0.5, 0.8])
    a = folktables_case.simulate_policy(scores, threshold=0.5)
    assert np.array_equal(a, np.array([0, 0, 1, 1]))


def test_lalonde_training_partition_selected() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["nsw_treated"] * 60 + ["nsw_control"] * 60 + ["cps_control"] * 80,
            "treat": [1] * 60 + [0] * 60 + [0] * 80,
            "re78": np.linspace(0, 1, 200),
            "re74": np.linspace(1, 2, 200),
            "age": np.linspace(20, 40, 200),
        }
    )
    selected = lalonde_case._select_training_partition(df)
    assert selected.shape[0] == 120
    assert set(selected["sample_id"].unique()) == {"nsw_treated", "nsw_control"}


def test_amazon_case_run_with_monkeypatched_loader(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(12)
    texts = np.array([f"sample review {i}" for i in range(1200)])
    labels = rng.binomial(1, 0.5, size=1200)
    domains = np.where(np.arange(1200) % 2 == 0, "books", "electronics")

    def fake_load_reviews(dataset_id=None, sample_size=50000):  # noqa: ARG001
        return {"text": texts, "label": labels, "domain": domains}

    monkeypatch.setattr(amazon_reviews_case, "load_reviews", fake_load_reviews)
    out_path = tmp_path / "amazon_metrics.json"
    metrics = amazon_reviews_case.run_case(
        output_path=out_path,
        sample_size=2000,
        max_rows=800,
        svd_dim=0,
    )
    assert "delta_propensity" in metrics
    assert metrics["negative_control"]["available"] is False
    assert isinstance(metrics["negative_control"]["reason"], str)
    assert "metric_sensitivity" in metrics
    assert out_path.exists()
