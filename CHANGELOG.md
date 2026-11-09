# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-09

### Fixed (numerical outputs change)

- `policy_regret_upper_bound`: transfer term is now `2 * k * M * delta` with `k = n_actions` for
  covariate-dependent policies and `k = 1` for fixed actions (new arguments `n_actions`, `policy_type`).
  The previous `M * delta` does not bound the regret (manuscript, Policy Regret Transfer theorem; the factor 2 is necessary).
- `policy_regret_minimax_floor`: `0.25 * M * delta` (previously `0.5 * M * delta`, which is false; 1/4 is sharp).
- `policy_regret_vc_bound`: corrected finite-sample bound for IPW empirical welfare maximization,
  `4 M delta + c (M / eta) (sqrt(v / n) + sqrt(log(1/xi) / n))`; `eta` is now required and `regret_observational` is ignored.
- `rkhs_rate_bound`: now `(n * eta) ** (-beta / (2 beta + d_w + d_y))` with new argument `d_y`; `xi` unused.

These mirror causaldef 0.3.0 and the corrected manuscript (concept DOI 10.5281/zenodo.18367346).

## [0.1.0] - 2026-02-16

### Added

- Core `AEGISMonitor` class with streaming window-based monitoring and threshold gating.
- Propensity-overlap proxy (`deficiency_proxy_propensity`) with `tv`, `w1`, and `mmd` metrics.
- Uniform deficiency proxy for discrete action sets.
- Negative-control IPTW discrepancy with configurable distance metrics.
- Negative-control sensitivity bound (`kappa * delta_nc`).
- Policy regret transfer bounds (`policy_regret_upper_bound`, `policy_regret_minimax_floor`).
- VC-augmented policy bound and RKHS rate bound utilities.
- Overlap diagnostic, confounding frontier, partial identification, and sharp lower bound helpers.
- Wasserstein deficiency for Gaussian distributions.
- Unit test suite with pytest.
- Synthetic validation simulation harness.
- Real-data case studies (LaLonde, Folktables, Amazon Reviews).
- GitHub Actions CI for Python 3.10, 3.11, 3.12.
