import numpy as np
import pytest

from aegis import (
    AEGISMonitor,
    confounding_frontier,
    deficiency_proxy_propensity,
    fit_logistic_newton,
    negative_control_distance,
    negative_control_sensitivity_bound,
    negative_control_tv,
    overlap_diagnostic,
    partial_id_set,
    policy_regret_vc_bound,
    policy_regret_minimax_floor,
    policy_regret_upper_bound,
    predict_propensity,
    rkhs_rate_bound,
    sharp_lower_bound,
    uniform_deficiency_proxy,
    wasserstein_deficiency_gaussian,
)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def test_fit_logistic_newton_recovers_signal() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 2))
    true_coef = np.array([-0.2, 1.0, -0.7])  # intercept + 2 features
    logits = true_coef[0] + x @ true_coef[1:]
    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p, size=n)

    coef_hat = fit_logistic_newton(x, y, l2=1e-3, max_iter=50, tol=1e-7)
    p_hat = predict_propensity(x, coef_hat)
    auc = _auc(y, p_hat)
    assert auc > 0.75
    assert np.sign(coef_hat[1]) == np.sign(true_coef[1])
    assert np.sign(coef_hat[2]) == np.sign(true_coef[2])


def test_deficiency_proxy_propensity_monotone_in_dependence_strength() -> None:
    rng = np.random.default_rng(1)
    n = 12000
    w = rng.normal(size=(n, 2))

    a_indep = rng.binomial(1, 0.5, size=n)
    delta_indep, _ = deficiency_proxy_propensity(w, a_indep, bins=40)

    p_strong = 1.0 / (1.0 + np.exp(-4.0 * w[:, 0]))
    a_strong = rng.binomial(1, p_strong, size=n)
    delta_strong, _ = deficiency_proxy_propensity(w, a_strong, bins=40)

    assert 0.0 <= delta_indep <= 1.0
    assert 0.0 <= delta_strong <= 1.0
    assert delta_strong > delta_indep


def test_negative_control_tv_flags_violation() -> None:
    rng = np.random.default_rng(2)
    n = 15000
    w = rng.normal(size=(n, 2))
    p = 1.0 / (1.0 + np.exp(-2.5 * w[:, 0]))
    a = rng.binomial(1, p, size=n)
    _, e = deficiency_proxy_propensity(w, a, bins=50)

    y_nc_ok = rng.normal(size=n)  # independent
    delta_ok = negative_control_tv(y_nc_ok, a, e, bins=50)

    y_nc_bad = 0.8 * a + rng.normal(size=n)
    delta_bad = negative_control_tv(y_nc_bad, a, e, bins=50)

    assert delta_bad > delta_ok


def test_metric_alternatives_for_propensity_proxy() -> None:
    rng = np.random.default_rng(21)
    n = 7000
    w = rng.normal(size=(n, 2))
    a_weak = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * w[:, 0])), size=n)
    a_strong = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.0 * w[:, 0])), size=n)

    for metric in ("tv", "w1", "mmd"):
        d_weak, _ = deficiency_proxy_propensity(w, a_weak, metric=metric, bins=40)
        d_strong, _ = deficiency_proxy_propensity(w, a_strong, metric=metric, bins=40)
        assert 0.0 <= d_weak <= 1.0
        assert 0.0 <= d_strong <= 1.0
        assert d_strong > d_weak


def test_negative_control_distance_alternatives() -> None:
    rng = np.random.default_rng(22)
    n = 9000
    w = rng.normal(size=(n, 2))
    p = 1.0 / (1.0 + np.exp(-2.0 * w[:, 0]))
    a = rng.binomial(1, p, size=n)
    _, e = deficiency_proxy_propensity(w, a, bins=40)

    y_nc_ok = rng.normal(size=n)
    y_nc_bad = 0.7 * a + rng.normal(size=n)

    for metric in ("tv", "w1", "mmd"):
        d_ok = negative_control_distance(y_nc_ok, a, e, metric=metric, bins=40)
        d_bad = negative_control_distance(y_nc_bad, a, e, metric=metric, bins=40)
        assert 0.0 <= d_ok <= 1.0
        assert 0.0 <= d_bad <= 1.0
        assert d_bad > d_ok


def test_monitor_trims_window() -> None:
    rng = np.random.default_rng(3)
    mon = AEGISMonitor(window_size=100, bins=20)
    w1 = rng.normal(size=(60, 3))
    a1 = rng.binomial(1, 0.5, size=60)
    mon.update(w1, a1)

    w2 = rng.normal(size=(60, 3))
    a2 = rng.binomial(1, 0.5, size=60)
    mon.update(w2, a2)

    assert mon._count == 100  # window trimmed
    metrics = mon.estimate()
    assert "delta_propensity" in metrics
    assert 0.0 <= metrics["delta_propensity"] <= 1.0


def test_rejects_non_binary_actions() -> None:
    mon = AEGISMonitor(window_size=10)
    w = np.zeros((3, 2))
    a = np.array([0, 2, 1])
    with pytest.raises(ValueError):
        mon.update(w, a)


def test_uniform_deficiency_proxy_multiaction() -> None:
    rng = np.random.default_rng(5)
    n = 5000
    w = rng.normal(size=(n, 2))
    logits0 = 1.0 * w[:, 0]
    logits1 = 1.0 * w[:, 1]
    logits2 = -0.8 * w[:, 0] - 0.5 * w[:, 1]
    logits = np.stack([logits0, logits1, logits2], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    draws = np.array([rng.choice(3, p=p) for p in probs], dtype=int)

    delta_uniform, per_action = uniform_deficiency_proxy(w, draws, bins=30)
    assert 0.0 <= delta_uniform <= 1.0
    assert set(per_action.keys()) == {"0", "1", "2"}
    assert all(0.0 <= d <= 1.0 for d in per_action.values())


def test_negative_control_bound_and_policy_bounds() -> None:
    assert negative_control_sensitivity_bound(0.2, kappa=1.5) == pytest.approx(0.3)
    assert policy_regret_upper_bound(0.1, 0.2, utility_range=(0.0, 2.0)) == pytest.approx(0.5)
    assert policy_regret_minimax_floor(0.2, utility_range=(0.0, 2.0)) == pytest.approx(0.2)


def test_assess_thresholds_and_decision_log() -> None:
    rng = np.random.default_rng(11)
    n = 4000
    w = rng.normal(size=(n, 2))
    p = 1.0 / (1.0 + np.exp(-4.0 * w[:, 0]))
    a = rng.binomial(1, p, size=n)

    monitor = AEGISMonitor(
        window_size=n,
        bins=40,
        delta_propensity_max=0.2,
        missing_nc_status="review",
    )
    monitor.update(w, a)
    decision = monitor.assess()
    assert decision["status"] in {"abstain", "review", "allow"}
    assert decision["status"] == "abstain"
    assert len(monitor.decision_log()) == 1


def test_assess_missing_negative_control_status_review() -> None:
    rng = np.random.default_rng(14)
    n = 2000
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, 0.5, size=n)

    monitor = AEGISMonitor(
        window_size=n,
        bins=30,
        delta_nc_max=0.2,
        missing_nc_status="review",
    )
    monitor.update(w, a)
    decision = monitor.assess()
    assert decision["status"] == "review"
    assert "delta_nc_missing" in decision["reasons"]


def test_monitor_with_w1_and_mmd_metrics() -> None:
    rng = np.random.default_rng(33)
    n = 2200
    w = rng.normal(size=(n, 3))
    p = 1.0 / (1.0 + np.exp(-1.4 * w[:, 0]))
    a = rng.binomial(1, p, size=n)
    y_nc = 0.3 * a + rng.normal(size=n)

    mon_w1 = AEGISMonitor(window_size=n, metric_propensity="w1", metric_nc="w1")
    mon_w1.update(w, a, y_nc=y_nc)
    m_w1 = mon_w1.estimate()
    assert 0.0 <= m_w1["delta_propensity"] <= 1.0
    assert 0.0 <= m_w1["delta_nc"] <= 1.0

    mon_mmd = AEGISMonitor(window_size=n, metric_propensity="mmd", metric_nc="mmd")
    mon_mmd.update(w, a, y_nc=y_nc)
    m_mmd = mon_mmd.estimate()
    assert 0.0 <= m_mmd["delta_propensity"] <= 1.0
    assert 0.0 <= m_mmd["delta_nc"] <= 1.0


def test_theory_utilities_from_causaldef_parity() -> None:
    assert rkhs_rate_bound(n=1000, beta=1.0, d_w=3, eta=0.2, xi=0.05) > 0
    assert wasserstein_deficiency_gaussian(alpha=1.0, gamma=0.5, sigma_a=1.0, a=1.0) > 0
    sharp = sharp_lower_bound(alpha=0.8, gamma=0.7, metric="tv")
    assert sharp["lower"] >= 0
    pid = partial_id_set(estimate=0.2, delta=0.1, estimand="ate", outcome_range=(0.0, 1.0))
    assert pid["upper"] > pid["lower"]

    vc = policy_regret_vc_bound(
        regret_observational=0.1,
        delta_uniform=0.2,
        vc_dim=5,
        n=1000,
        utility_range=(0.0, 1.0),
    )
    assert vc["regret_upper_bound"] > vc["transfer_penalty"]

    rng = np.random.default_rng(44)
    x = rng.normal(size=(800, 2))
    a = rng.binomial(1, 0.5, size=800)
    ov = overlap_diagnostic(x, a, trim=0.05)
    assert 0 <= ov["extreme_n"] <= ov["n"]

    frontier = confounding_frontier(grid_size=7)
    assert frontier["delta_grid"].shape == (7, 7)
