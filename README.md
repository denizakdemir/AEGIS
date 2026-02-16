# AEGIS — Adaptive Experiment-Gap Inference System

[![CI](https://github.com/denizakdemir/AEGIS/actions/workflows/ci.yml/badge.svg)](https://github.com/denizakdemir/AEGIS/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18601223.svg)](https://doi.org/10.5281/zenodo.18601223)

**AEGIS** is a lightweight Python package for monitoring the observational–interventional gap in deployed ML systems. It provides fast, streaming proxies of causal deficiency grounded in Le Cam's theory of experiment comparison — enabling pre-deployment audits, runtime monitoring, and per-decision gating with minimal dependencies.

## Features

- **Propensity-overlap proxy** — binary-action overlap diagnostic with metric choices `tv`, `w1`, or `mmd`.
- **Negative-control proxy** — IPTW negative-control discrepancy with configurable distance metrics.
- **Sensitivity bounds** — `kappa * delta_nc` upper bound for observable sensitivity.
- **Policy transfer bounds** — regret upper bounds, minimax floors, and VC-augmented policy bounds.
- **Streaming monitor** — `AEGISMonitor` with fixed-window updates, threshold-based gating, and decision logs.
- **Theory-aligned utilities** — overlap diagnostics, confounding frontier, partial identification sets, sharp lower bounds, and more.

## Installation

```bash
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[real-data]"   # Real-data case study dependencies
pip install -e ".[plots]"       # Matplotlib for validation plots
```

## Quickstart

```python
import numpy as np
from aegis import AEGISMonitor

rng = np.random.default_rng(7)
n = 1000
w = rng.normal(size=(n, 3))
a = rng.binomial(1, p=0.5, size=n)
y_nc = w[:, 0] + rng.normal(size=n)

monitor = AEGISMonitor(
    window_size=5000,
    bins=50,
    metric_propensity="w1",
    metric_nc="w1",
    delta_propensity_max=0.20,
    delta_nc_max=0.15,
    kappa=1.0,
)
monitor.update(w, a, y_nc=y_nc)

metrics = monitor.estimate()
decision = monitor.assess()

print(metrics)
print(decision["status"], decision["reasons"])
```

## Core API

| Function / Class | Description |
|---|---|
| `AEGISMonitor` | Streaming window monitor with threshold-based `assess()` and decision log |
| `deficiency_proxy_propensity(...)` | Binary-action overlap proxy |
| `uniform_deficiency_proxy(...)` | One-vs-rest uniform proxy over discrete action sets |
| `negative_control_distance(...)` | IPTW negative-control discrepancy (`tv` / `w1` / `mmd`) |
| `negative_control_sensitivity_bound(...)` | Compute `kappa * delta_nc` |
| `policy_regret_upper_bound(...)` | Regret ≤ Regret_obs + M × δ_uniform |
| `policy_regret_minimax_floor(...)` | 0.5 × M × δ_uniform |
| `overlap_diagnostic(...)` | ESS and extremes-based overlap diagnostic |
| `confounding_frontier(...)` | Confounding frontier utility |
| `sharp_lower_bound(...)` | Sharp two-point lower bound |
| `partial_id_set(...)` | Partial identification interval |

## Real-Data Case Studies

Example scripts are included in [`real_data/`](real_data/):

- `lalonde_case.py` — LaLonde job training program
- `folktables_case.py` — ACS/Folktables income data
- `amazon_reviews_case.py` — Amazon product reviews
- `run_all.py` — Run all case studies

```bash
python real_data/run_all.py --out-dir real_data/out
```

## Validation

Run the test suite:

```bash
pytest
```

Run synthetic validation simulations:

```bash
python validation/run_simulations.py --out-dir validation/out
```

Generate validation plots:

```bash
python validation/generate_plots.py --out-dir validation/out
```

## Repository Structure

```
AEGIS/
├── src/aegis/          # Installable package source
│   ├── __init__.py
│   ├── monitor.py      # Core monitor and propensity/NC proxies
│   ├── bounds.py       # Policy transfer and sensitivity bounds
│   ├── theory.py       # Theory-aligned utilities
│   ├── distances.py    # Distance metric implementations
│   └── version.py
├── tests/              # Unit tests
├── validation/         # Simulation-based validation harness
├── real_data/          # Real-data case study scripts
├── .github/workflows/  # CI configuration
├── pyproject.toml      # Package configuration
├── LICENSE             # MIT License
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Submit a pull request

## Citation

If you use AEGIS in your research, please cite the underlying theory paper:

> Akdemir, D. (2026). *Constraints on Causal Inference as Experiment Comparison: A Framework for Identification, Transportability, and Policy Learning.* Zenodo. [https://doi.org/10.5281/zenodo.18601223](https://doi.org/10.5281/zenodo.18601223)

BibTeX:

```bibtex
@article{akdemir2026causal,
  title   = {Constraints on Causal Inference as Experiment Comparison:
             A Framework for Identification, Transportability, and
             Policy Learning},
  author  = {Akdemir, Deniz},
  year    = {2026},
  doi     = {10.5281/zenodo.18601223},
  url     = {https://doi.org/10.5281/zenodo.18601223},
  publisher = {Zenodo}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Deniz Akdemir** — [deniz.akdemir.work@gmail.com](mailto:deniz.akdemir.work@gmail.com)
