"""Amazon multi-domain sentiment case study.

Loads a public multi-domain sentiment dataset from Hugging Face, builds a TF-IDF
representation, trains a classifier, and evaluates AEGIS metrics across domains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


def load_reviews(dataset_id: Optional[str] = None, sample_size: int = 50000) -> Dict[str, np.ndarray]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError("datasets is not installed") from exc

    candidates = [dataset_id] if dataset_id else [
        "amazon_reviews_multi",
        "amazon_polarity",
        "multi_domain_sentiment",
    ]

    text_candidates = ["review_body", "content", "text", "sentence", "title"]
    label_candidates = ["label", "stars", "sentiment", "polarity"]
    domain_candidates = ["product_category", "domain", "category", "reviewerID"]

    def first_present(df, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    for name in candidates:
        if not name:
            continue
        try:
            split = f"train[:{sample_size}]" if sample_size else "train"
            dataset = load_dataset(name, split=split, trust_remote_code=False)
        except Exception:
            continue

        df = dataset.to_pandas()
        text_col = first_present(df, text_candidates)
        label_col = first_present(df, label_candidates)
        domain_col = first_present(df, domain_candidates)

        if text_col is None or label_col is None:
            continue

        columns = [text_col, label_col] + ([domain_col] if domain_col in df.columns else [])
        df = df[columns].copy()
        df[text_col] = df[text_col].astype(str)

        if df[label_col].dtype.kind in {"i", "u", "f"}:
            unique_vals = df[label_col].unique()
            if unique_vals.size <= 2:
                df[label_col] = df[label_col].astype(int)
            else:
                df[label_col] = (df[label_col] > df[label_col].median()).astype(int)
        else:
            df[label_col] = df[label_col].astype("category").cat.codes

        if df[label_col].nunique() < 2:
            continue

        if domain_col not in df.columns:
            df["domain"] = "all"
            domain_col = "domain"

        return {
            "text": df[text_col].to_numpy(),
            "label": df[label_col].to_numpy(),
            "domain": df[domain_col].to_numpy(),
        }

    raise RuntimeError("Unable to load a usable sentiment dataset from Hugging Face")


def run_case(
    dataset_id: Optional[str] = None,
    output_path: Optional[Path] = None,
    sample_size: int = 50000,
    max_rows: int = 5000,
    svd_dim: int = 0,
    bootstrap_reps: int = 20,
) -> Dict[str, float]:
    data = load_reviews(dataset_id=dataset_id, sample_size=sample_size)
    texts = data["text"]
    labels = data["label"]
    domains = data["domain"]

    x_train, x_test, y_train, y_test, domain_train, domain_test = train_test_split(
        texts, labels, domains, test_size=0.3, random_state=42, stratify=domains
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, min_df=5)),
            ("clf", LogisticRegression(max_iter=200, n_jobs=1)),
        ]
    )
    model.fit(x_train, y_train)

    scores = model.predict_proba(x_test)[:, 1]
    a = (scores >= 0.5).astype(int)

    # Use a low-dimensional representation for W. By default we use the model
    # score itself (svd_dim=0), which is fast and conservative when A is a
    # thresholded score. Optionally compress sparse TF-IDF via SVD.
    tfidf = model.named_steps["tfidf"].transform(x_test)
    if tfidf.shape[0] > max_rows:
        idx = np.random.default_rng(7).choice(tfidf.shape[0], size=max_rows, replace=False)
        tfidf = tfidf[idx]
        scores = scores[idx]
        a = a[idx]
        domain_test = domain_test[idx]

    n_components = min(svd_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if n_components >= 2:
        w = TruncatedSVD(n_components=n_components, random_state=7).fit_transform(tfidf)
    else:
        w = scores.reshape(-1, 1)

    overall = summarize_case(
        w,
        a,
        y_nc=None,
        bins=50,
        bootstrap_reps=bootstrap_reps,
        random_state=7,
        missing_nc_reason=(
            "No validated negative-control outcome is available in the sentiment benchmark setup."
        ),
    )

    results = {"overall": overall}
    for dom in np.unique(domain_test):
        mask = domain_test == dom
        if mask.sum() < 50:
            continue
        monitor = AEGISMonitor(window_size=min(5000, mask.sum()), bins=50)
        monitor.update(w[mask], a[mask])
        dom_metrics = monitor.estimate()
        dom_metrics["negative_control"] = {
            "available": False,
            "reason": "No validated negative-control outcome is available in the sentiment benchmark setup.",
        }
        results[str(dom)] = dom_metrics

    if output_path is not None:
        save_metrics(output_path, results)

    return overall


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Amazon reviews AEGIS case study")
    parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset id")
    parser.add_argument("--out", type=Path, default=Path("amazon_metrics.json"))
    parser.add_argument("--sample-size", type=int, default=50000, help="Max rows to load from HF train split")
    parser.add_argument("--max-rows", type=int, default=5000, help="Max rows used for AEGIS evaluation")
    parser.add_argument(
        "--svd-dim",
        type=int,
        default=0,
        help="Dimension for TF-IDF SVD compression (0 uses model score as W).",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=20, help="Bootstrap repetitions for CI")
    args = parser.parse_args()

    metrics = run_case(
        dataset_id=args.dataset,
        output_path=args.out,
        sample_size=args.sample_size,
        max_rows=args.max_rows,
        svd_dim=args.svd_dim,
        bootstrap_reps=args.bootstrap_reps,
    )
    print(metrics)
