# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
