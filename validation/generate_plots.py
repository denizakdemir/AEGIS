"""Generate manuscript-ready plots for AEGIS validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required for plot generation. Install with: pip install matplotlib"
    ) from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis import AEGISMonitor, confounding_frontier, deficiency_proxy_propensity  # noqa: E402
from aegis.distances import distance_1d  # noqa: E402
from simulations import gating_performance, negative_control_curve, overlap_curve, run_validation, uniform_curve  # noqa: E402


METRIC_COLORS = {"tv": "#1b9e77", "w1": "#d95f02", "mmd": "#7570b3"}
STATUS_COLORS = {"allow": "#1b9e77", "review": "#e6ab02", "abstain": "#d95f02"}


def _style_axes(ax) -> None:
    ax.grid(alpha=0.25, linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)


def _load_summary(out_dir: Path, n_scale: float) -> dict:
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return run_validation(out_dir=out_dir, n_scale=n_scale)


def _plot_synthetic_curves(summary: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)

    curves = [
        ("overlap_curve", "Overlap Proxy (TV)", r"$\widehat{\delta}_{prop}$", "#1f77b4"),
        ("negative_control_curve", "Negative-Control Proxy (TV)", r"$\widehat{\delta}_{NC}$", "#2ca02c"),
        ("uniform_curve", "Uniform Proxy (TV)", r"$\widehat{\delta}_{uniform}$", "#9467bd"),
    ]
    for ax, (key, title, ylab, color) in zip(axes, curves):
        points = summary[key]
        x = np.asarray([p["strength"] for p in points], dtype=float)
        y = np.asarray([p["mean"] for p in points], dtype=float)
        s = np.asarray([p["std"] for p in points], dtype=float)
        ci = 1.96 * s
        lo = np.clip(y - ci, 0.0, 1.0)
        hi = np.clip(y + ci, 0.0, 1.0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)
        ax.errorbar(x, y, yerr=s, marker="o", linewidth=2, color=color, capsize=3)
        ax.set_title(title)
        ax.set_xlabel("Confounding Strength")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1)
        ax.axhline(0.20, linestyle="--", color="black", alpha=0.45, linewidth=1)
        ax.text(
            x[-1],
            y[-1] + 0.04,
            f"{y[0]:.3f} -> {y[-1]:.3f}",
            fontsize=8,
            ha="right",
            va="bottom",
            color=color,
        )
        _style_axes(ax)

    fig.suptitle("Synthetic Validation Curves (TV): Means +/- SD with 95% CI Band", y=1.04, fontsize=12)
    fig.savefig(fig_dir / "synthetic_curves_tv.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_comparison(fig_dir: Path, fast: bool) -> None:
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
    metrics = ["tv", "w1", "mmd"]

    n_curve = 2500 if fast else 5500
    repeats = 4 if fast else 6

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    panel_defs = [
        ("Overlap Proxy", overlap_curve, r"$\widehat{\delta}_{prop}$"),
        ("Negative-Control Proxy", negative_control_curve, r"$\widehat{\delta}_{NC}$"),
        ("Uniform Proxy", uniform_curve, r"$\widehat{\delta}_{uniform}$"),
    ]

    for ax, (title, fn, ylab) in zip(axes, panel_defs):
        for i, metric in enumerate(metrics):
            points = fn(
                strengths,
                n=n_curve,
                repeats=repeats,
                seed=100 + i,
                metric=metric,
            )
            x = np.asarray([p.strength for p in points], dtype=float)
            y = np.asarray([p.mean for p in points], dtype=float)
            s = np.asarray([p.std for p in points], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.2, label=metric.upper(), color=METRIC_COLORS[metric])
            ax.fill_between(
                x,
                np.clip(y - s, 0.0, 1.0),
                np.clip(y + s, 0.0, 1.0),
                color=METRIC_COLORS[metric],
                alpha=0.12,
                linewidth=0,
            )
            ax.text(
                x[-1] + 0.04,
                y[-1],
                f"{y[-1]:.2f}",
                color=METRIC_COLORS[metric],
                fontsize=8,
                va="center",
            )
        ax.set_title(title)
        ax.set_xlabel("Confounding Strength")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1)
        ax.axhline(0.20, linestyle="--", color="black", alpha=0.4, linewidth=1)
        _style_axes(ax)

    axes[2].legend(loc="lower right", frameon=True, title="Distance")
    fig.suptitle("Metric Comparison: Sensitivity Profile Depends on Geometry", y=1.04, fontsize=12)
    fig.savefig(fig_dir / "metric_comparison_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_gating_heatmaps(fig_dir: Path, fast: bool) -> None:
    tau_prop = np.linspace(0.10, 0.40, 9)
    tau_nc = np.linspace(0.10, 0.30, 9)

    n_windows = 10 if fast else 18
    window_size = 450 if fast else 650

    safe = np.zeros((tau_nc.size, tau_prop.size))
    risky = np.zeros((tau_nc.size, tau_prop.size))
    for i, tnc in enumerate(tau_nc):
        for j, tprop in enumerate(tau_prop):
            perf = gating_performance(
                n_windows=n_windows,
                window_size=window_size,
                seed=701,
                delta_propensity_max=float(tprop),
                delta_nc_max=float(tnc),
            )
            safe[i, j] = perf["safe_allow_rate"]
            risky[i, j] = perf["risky_abstain_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(safe, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="YlGn")
    axes[0].set_title("Safe Allow Rate (higher is better)")
    axes[0].set_xlabel(r"$\tau_{prop}$")
    axes[0].set_ylabel(r"$\tau_{NC}$")
    axes[0].set_xticks(range(tau_prop.size), [f"{v:.2f}" for v in tau_prop], rotation=45)
    axes[0].set_yticks(range(tau_nc.size), [f"{v:.2f}" for v in tau_nc])
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.046)
    c0.set_label("Rate")

    im1 = axes[1].imshow(risky, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="OrRd")
    axes[1].set_title("Risky Abstain Rate (higher is better)")
    axes[1].set_xlabel(r"$\tau_{prop}$")
    axes[1].set_ylabel(r"$\tau_{NC}$")
    axes[1].set_xticks(range(tau_prop.size), [f"{v:.2f}" for v in tau_prop], rotation=45)
    axes[1].set_yticks(range(tau_nc.size), [f"{v:.2f}" for v in tau_nc])
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.046)
    c1.set_label("Rate")

    j_star = int(np.argmin(np.abs(tau_prop - 0.20)))
    i_star = int(np.argmin(np.abs(tau_nc - 0.20)))
    for ax in axes:
        ax.scatter([j_star], [i_star], marker="x", s=80, c="black", linewidth=1.5, label="default (0.20, 0.20)")
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    axes[0].text(0.02, 0.02, f"default={safe[i_star, j_star]:.2f}", transform=axes[0].transAxes, fontsize=8)
    axes[1].text(0.02, 0.02, f"default={risky[i_star, j_star]:.2f}", transform=axes[1].transAxes, fontsize=8)

    fig.suptitle("Gating Threshold Surface: Trade-off Map for Governance Tuning", y=1.03, fontsize=12)
    fig.savefig(fig_dir / "gating_heatmaps.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_propensity_overlap_example(fig_dir: Path) -> None:
    rng = np.random.default_rng(811)
    n = 5000
    w = rng.normal(size=(n, 2))
    p = 1.0 / (1.0 + np.exp(-2.0 * w[:, 0]))
    a = rng.binomial(1, p, size=n)

    _, e = deficiency_proxy_propensity(w, a, metric="tv")
    e_t = e[a == 1]
    e_c = e[a == 0]

    d_tv = distance_1d(e_t, e_c, metric="tv", bins=50, value_range=(0.0, 1.0))
    d_w1 = distance_1d(e_t, e_c, metric="w1", value_range=(0.0, 1.0))
    d_mmd = distance_1d(e_t, e_c, metric="mmd")

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    bins = np.linspace(0, 1, 40)
    ax.hist(e_c, bins=bins, alpha=0.35, density=True, label=f"Control (n={e_c.size})", color="#377eb8")
    ax.hist(
        e_t,
        bins=bins,
        alpha=0.35,
        density=True,
        label=f"Treated (n={e_t.size})",
        color="#e41a1c",
    )
    ax.hist(e_c, bins=bins, histtype="step", linewidth=1.8, density=True, color="#1f4f7a")
    ax.hist(e_t, bins=bins, histtype="step", linewidth=1.8, density=True, color="#9d1414")
    ax.axvline(np.mean(e_c), color="#1f4f7a", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.axvline(np.mean(e_t), color="#9d1414", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Overlap Diagnostic (Synthetic Binary Action)")
    ax.legend()
    _style_axes(ax)
    ax.text(
        0.02,
        0.96,
        f"TV={d_tv:.3f}\nW1={d_w1:.3f}\nMMD={d_mmd:.3f}\nmean(control)={np.mean(e_c):.3f}\nmean(treated)={np.mean(e_t):.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.savefig(fig_dir / "propensity_overlap_example.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_streaming_timeline(fig_dir: Path, fast: bool) -> None:
    rng = np.random.default_rng(915)
    n_windows = 40 if fast else 70
    batch_size = 180 if fast else 220
    mon = AEGISMonitor(
        window_size=1000,
        bins=40,
        metric_propensity="w1",
        metric_nc="w1",
        delta_propensity_max=0.20,
        delta_nc_max=0.20,
        kappa=1.0,
    )

    dp, dn, status = [], [], []
    for t in range(n_windows):
        risk = 0.0 if t < n_windows // 2 else 1.0
        u = rng.normal(size=batch_size)
        w = rng.normal(size=(batch_size, 3))
        p = 1.0 / (1.0 + np.exp(-(0.4 + 1.6 * risk) * w[:, 0] - (0.2 + 1.2 * risk) * u))
        a = rng.binomial(1, p, size=batch_size)
        y_nc = (0.2 + 0.8 * risk) * u + rng.normal(size=batch_size)

        mon.update(w, a, y_nc=y_nc)
        decision = mon.assess(record=False)
        metrics = decision["metrics"]
        dp.append(metrics["delta_propensity"])
        dn.append(metrics["delta_nc"])
        status.append(decision["status"])

    x = np.arange(n_windows)
    fig, (ax, ax_status) = plt.subplots(
        2,
        1,
        figsize=(10, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.2, 0.8]},
        constrained_layout=True,
    )
    ax.plot(x, dp, label=r"$\widehat{\delta}_{prop}$", linewidth=2.2, color="#1f77b4")
    ax.plot(x, dn, label=r"$\widehat{\delta}_{NC}$", linewidth=2.2, color="#ff7f0e")
    ax.axhline(0.20, linestyle="--", color="black", alpha=0.75, label="Threshold=0.20")
    ax.axvline(n_windows // 2, linestyle=":", color="gray", linewidth=1.5)
    ax.text((n_windows // 4), 0.93, "low-risk regime", fontsize=8, color="gray")
    ax.text((3 * n_windows // 4), 0.93, "high-risk regime", fontsize=8, color="gray", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proxy Value")
    ax.set_title("Streaming Monitor Timeline with Regime Shift")
    _style_axes(ax)
    ax.legend(ncol=3, fontsize=8, loc="upper left", frameon=True)

    status_codes = np.array([{"allow": 0, "review": 1, "abstain": 2}[s] for s in status], dtype=int)
    cmap = mcolors.ListedColormap([STATUS_COLORS["allow"], STATUS_COLORS["review"], STATUS_COLORS["abstain"]])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax_status.imshow(status_codes.reshape(1, -1), aspect="auto", cmap=cmap, norm=norm, origin="lower")
    ax_status.set_yticks([])
    ax_status.set_ylabel("Gate", rotation=0, labelpad=18, va="center")
    ax_status.set_xlabel("Window Index")
    ax_status.axvline(n_windows // 2, linestyle=":", color="gray", linewidth=1.2)

    legend_handles = [
        mpatches.Patch(color=STATUS_COLORS["allow"], label="allow"),
        mpatches.Patch(color=STATUS_COLORS["review"], label="review"),
        mpatches.Patch(color=STATUS_COLORS["abstain"], label="abstain"),
    ]
    ax_status.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.35),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.savefig(fig_dir / "streaming_timeline.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_confounding_frontier(fig_dir: Path) -> None:
    frontier = confounding_frontier(alpha_range=(-2.0, 2.0), gamma_range=(-2.0, 2.0), grid_size=41)
    alpha_grid = frontier["alpha_grid"]
    gamma_grid = frontier["gamma_grid"]
    delta_grid = frontier["delta_grid"]

    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    im = ax.imshow(
        delta_grid,
        origin="lower",
        extent=(alpha_grid.min(), alpha_grid.max(), gamma_grid.min(), gamma_grid.max()),
        aspect="auto",
        cmap="viridis",
    )
    contours = ax.contour(alpha_grid, gamma_grid, delta_grid, levels=[0.01, 0.05, 0.10, 0.20], colors="white", linewidths=1.2)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
    ax.axhline(0.0, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(0.0, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(r"$\alpha$ (U $\to$ A)")
    ax.set_ylabel(r"$\gamma$ (U $\to$ Y)")
    ax.set_title("Confounding Frontier (Linear-Gaussian Lower Bound)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(r"Lower bound on $\delta$")
    ax.text(-1.9, 1.75, "stronger confounding", color="white", fontsize=8)
    fig.savefig(fig_dir / "confounding_frontier.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_kappa_sensitivity(summary: dict, fig_dir: Path) -> None:
    delta_nc_ref = summary["negative_control_curve"][2]["mean"]
    kappas = np.linspace(0.5, 3.0, 80)
    bounds = np.clip(kappas * delta_nc_ref, 0, 1)

    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    ax.plot(kappas, bounds, linewidth=2.2, color="#1f77b4")
    ax.axvline(1.0, linestyle="--", color="black", alpha=0.7, linewidth=1.2, label=r"$\kappa=1$")
    ax.axhline(delta_nc_ref, linestyle=":", color="#444444", alpha=0.8, linewidth=1.2, label=r"$\widehat{\delta}_{NC}$ at strength=1.0")
    target = 0.20
    kappa_cross = target / max(delta_nc_ref, 1e-9)
    if kappas.min() <= kappa_cross <= kappas.max():
        ax.axhline(target, linestyle="--", color="#d95f02", alpha=0.7, linewidth=1.2)
        ax.axvline(kappa_cross, linestyle="--", color="#d95f02", alpha=0.7, linewidth=1.2)
        ax.text(
            kappa_cross + 0.02,
            target + 0.03,
            rf"$\kappa \approx {kappa_cross:.2f}$ for bound={target:.2f}",
            fontsize=8,
            color="#d95f02",
        )
    ax.set_xlabel(r"Sensitivity Parameter $\kappa$")
    ax.set_ylabel(r"Upper Bound $\kappa \widehat{\delta}_{NC}$")
    ax.set_ylim(0, 1)
    ax.set_title("Negative-Control Sensitivity Curve")
    _style_axes(ax)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.savefig(fig_dir / "kappa_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(out_dir: Path, *, fast: bool = False) -> dict:
    n_scale = 0.35 if fast else 1.0
    summary = _load_summary(out_dir=out_dir, n_scale=n_scale)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_synthetic_curves(summary, fig_dir)
    _plot_metric_comparison(fig_dir, fast=fast)
    _plot_gating_heatmaps(fig_dir, fast=fast)
    _plot_propensity_overlap_example(fig_dir)
    _plot_streaming_timeline(fig_dir, fast=fast)
    _plot_confounding_frontier(fig_dir)
    _plot_kappa_sensitivity(summary, fig_dir)

    manifest = {
        "figures": sorted([p.name for p in fig_dir.glob("*.png")]),
        "source_summary": str((out_dir / "summary.json").resolve()),
    }
    (fig_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AEGIS validation plots")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation/out"),
        help="Validation output directory (must contain or generate summary.json)",
    )
    parser.add_argument("--fast", action="store_true", help="Use reduced simulation budget.")
    args = parser.parse_args()

    manifest = generate_all_plots(args.out_dir, fast=args.fast)
    print(f"[OK] Wrote {len(manifest['figures'])} figures to {args.out_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
