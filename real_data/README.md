# AEGIS Real-Data Case Studies

This folder contains runnable reference scripts for the manuscript case studies.

## Case Studies

### 1) Lalonde Job Training (NSW)
- Source: public versions via R `MatchIt` or Python `causaldata`.
- Action `A`: treatment assignment (job training).
- Outcome `Y`: post-treatment earnings.
- Negative control: pre-treatment earnings.
- Script: `lalonde_case.py`

### 2) Folktables ACS Income
- Source: `folktables` Python package.
- Action `A`: simulated policy decision from a classifier.
- Outcome `Y`: income threshold.
- Negative control: pre-decision covariates (diagnostic only).
- Script: `folktables_case.py`

### 3) Amazon Multi-Domain Sentiment (optional)
- Source: public multi-domain sentiment datasets.
- Action `A`: moderation decision from a sentiment classifier.
- Outcome `Y`: sentiment label.
- Script: `amazon_reviews_case.py`

## Expected Outputs

Each script emits:
- `delta_propensity` and optional `delta_nc`.
- `uncertainty` block with bootstrap summaries (single-run estimate, mean/std, 95% CI).
- `metric_sensitivity` block for `tv`, `w1`, and `mmd`.
- explicit `negative_control` availability metadata (`available`, `reason`).
- A short JSON or CSV summary for inclusion in the paper.
- A consolidated `real_data_summary.json` when using `run_all.py`.
- Manuscript-ready table artifacts:
  - `real_data_proxy_table.csv`
  - `real_data_proxy_table.md`

## Dependencies

From `AEGIS/`, install optional dependencies:

```bash
pip install -e ".[real-data]"
```

## Run examples

```bash
python AEGIS/real_data/lalonde_case.py --out AEGIS/real_data/out/lalonde_metrics.csv
python AEGIS/real_data/folktables_case.py --states CA NY TX --out AEGIS/real_data/out/folktables_metrics.json
python AEGIS/real_data/amazon_reviews_case.py --out AEGIS/real_data/out/amazon_metrics.json
```

## Run everything (best-effort)

```bash
python AEGIS/real_data/run_all.py --out-dir AEGIS/real_data/out
```

Notes:
- The Amazon script defaults to using the model score as the monitored covariate (`--svd-dim 0`), which is fast and conservative when the action is a thresholded score. Use `--svd-dim 50` to monitor overlap in a low-dimensional TF-IDF representation instead.
- If `causaldata` is unavailable, pass a local Lalonde/NSW CSV:
  - `python AEGIS/real_data/run_all.py --out-dir AEGIS/real_data/out --lalonde-data /path/to/nsw.csv`
