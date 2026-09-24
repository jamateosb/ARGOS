#!/usr/bin/env python3
"""Generate every public ARGOS figure from the published result tables.

Inputs are only the CSV tables under results/controlled/tables and
results/live/tables. Outputs go to results/figures/pdf (vector, for print)
and results/figures/png (raster previews).

    python scripts/generate_figures.py                 # all figures
    python scripts/generate_figures.py --main          # the eight main panels only

Figure conventions:
- no embedded titles; results/README.md describes each figure,
- one visual identity per controller across all figures,
- secondary encodings (tick labels, markers, line styles, hatching) so that no
  controller is identified by color alone,
- main panels sized for one column (3.45 in) of a two-column page.

After drawing, every figure's visible text is audited for internal identifiers
(for example ``best_fixed``) and for non-English labels before it is saved.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Fixed metadata timestamp so regenerated PDFs are byte-identical run to run.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1753315200")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
CONTROLLED = ROOT / "results/controlled/tables"
LIVE = ROOT / "results/live/tables"
OUT = ROOT / "results/figures"

CONTROLLER_ORDER = ["static", "threshold", "qlearning", "dqn", "ppo", "best_fixed"]
LEARNERS = ["qlearning", "dqn", "ppo"]
CONTROLLER_LABEL = {
    "static": "Static",
    "threshold": "Threshold",
    "qlearning": "Q-learning",
    "dqn": "DQN",
    "ppo": "PPO",
    "best_fixed": "Best-fixed",
}
CONTROLLER_COLOR = {
    "static": "#5b6472",
    "threshold": "#b07a33",
    "qlearning": "#7d4fa3",
    "dqn": "#1d9a86",
    "ppo": "#d5495e",
    "best_fixed": "#2679a3",
}
CONTROLLER_MARKER = {
    "static": "o",
    "threshold": "v",
    "qlearning": "D",
    "dqn": "s",
    "ppo": "^",
    "best_fixed": "P",
}
# Best-fixed is an offline-selected reference, not an online controller; the
# hatch marks it as such and separates it from Q-learning for deutan readers.
CONTROLLER_HATCH = {"best_fixed": "////"}
PROFILE_ORDER = [
    "lax-background",
    "short-burst",
    "aggressive-incident",
    "standard-operations",
    "cost-sensitive",
]
PROFILE_LABEL = {
    "lax-background": "Lax\nbackground",
    "short-burst": "Short\nburst",
    "aggressive-incident": "Aggressive\nincident",
    "standard-operations": "Standard\noperations",
    "cost-sensitive": "Cost\nsensitive",
}
RUNTIME_LABEL = {"thread": "Thread runtime", "process": "Process runtime"}
SCENARIO_LABEL = {"realistic": "Realistic regime", "concurrency": "Concurrency regime"}
ACTION_ORDER = [
    "hold",
    "increase_coverage",
    "decrease_coverage",
    "increase_sample",
    "decrease_sample",
    "increase_freshness",
    "decrease_freshness",
]
# Freshness is an update interval: increase_freshness shortens it (fresher
# results, more work) and decrease_freshness lengthens it (staler, cheaper).
ACTION_LABEL = {
    "hold": "Hold",
    "increase_coverage": "Coverage +",
    "decrease_coverage": "Coverage −",
    "increase_sample": "Sample +",
    "decrease_sample": "Sample −",
    "increase_freshness": "Fresher (interval −)",
    "decrease_freshness": "Staler (interval +)",
}
ACTION_COLOR = {
    "hold": "#d4d7dc",
    "increase_coverage": "#1f5f99",
    "decrease_coverage": "#8fb8de",
    "increase_sample": "#2e7d4f",
    "decrease_sample": "#9fd3b0",
    "increase_freshness": "#b35c1e",
    "decrease_freshness": "#f0b983",
}

SINGLE = 3.45  # inches, one column of a two-column page
DOUBLE = 7.0  # inches, full text width

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "hatch.linewidth": 0.6,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
    }
)

FORBIDDEN_TEXT = re.compile(
    r"best_fixed|qlearning|_|Best fixed|Best Fixed|sample size|"
    r"\b(semilla|perfil|acciones|peticiones|violaciones|recompensa|controlador|régimen|carga)\b",
    re.IGNORECASE,
)

FIGURES: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    if not path.is_file():
        sys.exit(f"Missing published table: {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def _t_interval(values: pd.Series) -> tuple[float, float]:
    """Mean and half-width of the two-sided Student-t 95% interval."""
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size < 2:
        return float(clean.mean()), 0.0
    half = t.ppf(0.975, clean.size - 1) * clean.std(ddof=1) / np.sqrt(clean.size)
    return float(clean.mean()), float(half)


def _visible_text(fig: plt.Figure) -> list[str]:
    texts = [text.get_text() for text in fig.findobj(matplotlib.text.Text)]
    return [text for text in texts if text.strip()]


def _save(fig: plt.Figure, name: str) -> None:
    offending = [text for text in _visible_text(fig) if FORBIDDEN_TEXT.search(text)]
    if offending:
        raise RuntimeError(f"{name}: internal or non-English label text {offending}")
    for fmt, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        folder = OUT / fmt
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.{fmt}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    FIGURES.append(name)


def _bar_style(controller: str) -> dict:
    return {
        "facecolor": CONTROLLER_COLOR[controller],
        "hatch": CONTROLLER_HATCH.get(controller),
        "edgecolor": "white",
        "linewidth": 0.6,
    }


def _controller_handles(controllers: list[str]) -> list[Patch]:
    return [Patch(label=CONTROLLER_LABEL[c], **_bar_style(c)) for c in controllers]


# ---------------------------------------------------------------------------
# Main figures
# ---------------------------------------------------------------------------


def controlled_mean_reward() -> None:
    """Mean frozen reward per decision; error bars are Student-t 95% intervals over five seed means."""
    rows = _read(CONTROLLED, "controlled_reward_summary")
    data = {(r.runtime, r.controller): r for r in rows.itertuples()}
    ymin = rows.ci95_low.min() - 0.03
    ymax = rows.ci95_high.max() + 0.03
    for runtime in ("thread", "process"):
        fig, ax = plt.subplots(figsize=(SINGLE, 2.55))
        x = np.arange(len(CONTROLLER_ORDER))
        means = [data[(runtime, c)].mean_reward for c in CONTROLLER_ORDER]
        low = [data[(runtime, c)].mean_reward - data[(runtime, c)].ci95_low for c in CONTROLLER_ORDER]
        high = [data[(runtime, c)].ci95_high - data[(runtime, c)].mean_reward for c in CONTROLLER_ORDER]
        for i, c in enumerate(CONTROLLER_ORDER):
            ax.bar(x[i], means[i], width=0.62, **_bar_style(c))
        ax.errorbar(x, means, yerr=[low, high], fmt="none", ecolor="black", capsize=3, linewidth=1.0, capthick=1.0)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([CONTROLLER_LABEL[c] for c in CONTROLLER_ORDER], rotation=32, ha="right")
        ax.set_ylabel("Mean reward per decision")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        fig.tight_layout()
        _save(fig, f"controlled_mean_reward_{runtime}")


def controlled_reward_by_profile() -> None:
    """Mean frozen reward per decision by workload profile; each bar averages five evaluation seeds."""
    rows = _read(CONTROLLED, "controlled_profile_reward_summary")
    data = {(r.runtime, r.profile, r.controller): r.mean_reward for r in rows.itertuples()}
    ymin = min(data.values()) - 0.03
    ymax = max(data.values()) + 0.03
    width = 0.13
    for runtime in ("thread", "process"):
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.45))
        for j, c in enumerate(CONTROLLER_ORDER):
            xs = [i + (j - 2.5) * width for i in range(len(PROFILE_ORDER))]
            ax.bar(xs, [data[(runtime, p, c)] for p in PROFILE_ORDER], width=width * 0.92, **_bar_style(c))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(range(len(PROFILE_ORDER)))
        ax.set_xticklabels([PROFILE_LABEL[p] for p in PROFILE_ORDER])
        ax.set_ylabel("Mean reward per decision")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        fig.legend(
            handles=_controller_handles(CONTROLLER_ORDER),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=6,
            frameon=False,
            columnspacing=1.1,
            handlelength=1.4,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _save(fig, f"controlled_reward_by_profile_{runtime}")


def live_request_load() -> None:
    """Active and queued requests, averaged over the five paired live seeds, on a 30 s grid."""
    rows = _read(LIVE, "live_load_timeline")
    mean = rows.groupby(["scenario", "variant", "minute"], as_index=False)[["active_requests", "queued_requests"]].mean()
    # The three variants share one admission trace by design; decreasing line
    # widths with a fixed draw order keep every variant visible where the
    # curves coincide instead of letting the last one painted occlude the rest.
    styles = {"static": 2.4, "dqn": 1.4, "ppo": 0.7}
    for scenario in ("realistic", "concurrency"):
        fig, ax = plt.subplots(figsize=(SINGLE, 2.75))
        for variant, width in styles.items():
            sub = mean[(mean.scenario == scenario) & (mean.variant == variant)]
            color, label = CONTROLLER_COLOR[variant], CONTROLLER_LABEL[variant]
            ax.plot(sub.minute, sub.active_requests, color=color, linewidth=width, label=f"{label} active",
                    solid_capstyle="round")
            ax.plot(sub.minute, sub.queued_requests, color=color, linewidth=width, linestyle="--",
                    label=f"{label} queued")
        ax.set_ylabel("Mean requests")
        ax.set_xlabel("Elapsed time (minutes)")
        ax.set_xlim(0, 30)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, linewidth=0.5)
        # Column-major legend: one column per variant, active above queued.
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, columnspacing=0.8,
                   handlelength=1.5, borderaxespad=0.1)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        _save(fig, f"live_request_load_{scenario}")


def live_reward_by_seed() -> None:
    """Mean decision reward per live trial, paired by live seed (identical arrival trace per seed)."""
    rows = _read(LIVE, "live_runs")
    ymin, ymax = rows.reward_per_step.min() - 0.04, rows.reward_per_step.max() + 0.04
    for scenario in ("realistic", "concurrency"):
        fig, ax = plt.subplots(figsize=(SINGLE, 2.55))
        seeds = sorted(rows.live_seed.unique())
        x = np.arange(len(seeds))
        for variant in ("static", "dqn", "ppo"):
            sub = rows[(rows.scenario == scenario) & (rows.variant == variant)].set_index("live_seed").loc[seeds]
            ax.plot(x, sub.reward_per_step, marker=CONTROLLER_MARKER[variant], color=CONTROLLER_COLOR[variant],
                    linewidth=1.5, markersize=4.5, label=CONTROLLER_LABEL[variant])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Live seed")
        ax.set_ylabel("Mean decision reward")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, columnspacing=1.0,
                   handlelength=1.4, borderaxespad=0.1)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        _save(fig, f"live_reward_by_seed_{scenario}")


# ---------------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------------


def controlled_paired_reward_deltas() -> None:
    """Paired reward deltas of each learned controller against each non-learning comparator."""
    rows = _read(CONTROLLED, "controlled_paired_reward_deltas")
    baselines = ["static", "threshold", "best_fixed"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), sharey=True)
    width = 0.24
    rng = np.random.default_rng(0)  # jitter only, for legibility of the seed dots
    for ax, runtime in zip(axes, ("thread", "process")):
        for j, learner in enumerate(LEARNERS):
            for i, baseline in enumerate(baselines):
                deltas = rows[(rows.runtime == runtime) & (rows.controller == learner) & (rows.baseline == baseline)]
                center, half = _t_interval(deltas.reward_delta)
                xpos = i + (j - 1) * width
                ax.bar(xpos, center, width=width * 0.9, **_bar_style(learner))
                ax.errorbar(xpos, center, yerr=half, fmt="none", ecolor="black", capsize=2.5, linewidth=0.9)
                jitter = rng.uniform(-0.05, 0.05, len(deltas))
                ax.scatter(xpos + jitter, deltas.reward_delta, s=7, color="black", alpha=0.55, linewidths=0, zorder=3)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(baselines)))
        ax.set_xticklabels([f"vs. {CONTROLLER_LABEL[b]}" for b in baselines])
        ax.set_title(RUNTIME_LABEL[runtime])
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Paired reward delta per decision")
    handles = _controller_handles(LEARNERS) + [
        Line2D([], [], marker="o", linestyle="none", color="black", alpha=0.55, markersize=3, label="Seed pair")
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "controlled_paired_reward_deltas")


def controlled_fidelity_vs_duty() -> None:
    """Delivered spatial fidelity against service duty, each point the mean of 25 frozen evaluations."""
    runs = _read(CONTROLLED, "controlled_runs")
    means = runs.groupby(["runtime", "controller"], as_index=False)[
        ["spatial_fidelity_mean", "service_duty_mean_percent"]
    ].mean()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.8), sharex=True, sharey=True)
    offsets = {
        "static": (6, -9),
        "threshold": (6, 3),
        "qlearning": (6, -9),
        "dqn": (-6, 5),
        "ppo": (6, -9),
        "best_fixed": (-6, -11),
    }
    for ax, runtime in zip(axes, ("thread", "process")):
        for c in CONTROLLER_ORDER:
            row = means[(means.runtime == runtime) & (means.controller == c)].iloc[0]
            ax.scatter(row.service_duty_mean_percent, row.spatial_fidelity_mean, s=46,
                       marker=CONTROLLER_MARKER[c], color=CONTROLLER_COLOR[c], edgecolors="white", linewidths=0.8,
                       zorder=3)
            dx, dy = offsets[c]
            ax.annotate(CONTROLLER_LABEL[c], (row.service_duty_mean_percent, row.spatial_fidelity_mean),
                        xytext=(dx, dy), textcoords="offset points", fontsize=8,
                        ha="right" if dx < 0 else "left")
        ax.set_title(RUNTIME_LABEL[runtime])
        ax.set_xlabel("Mean service duty (%)")
        ax.set_xlim(0, 21)
        ax.set_ylim(0.9, 0.955)
        ax.grid(alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Mean spatial fidelity")
    fig.tight_layout()
    _save(fig, "controlled_fidelity_vs_duty")


def controlled_resource_pressure() -> None:
    """Cluster CPU and memory 95th percentiles per run, averaged over 25 frozen evaluations."""
    runs = _read(CONTROLLED, "controlled_runs")
    means = runs.groupby(["runtime", "controller"])[["cpu_p95", "memory_p95"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), sharey=True)
    x = np.arange(len(CONTROLLER_ORDER))
    width = 0.38
    for ax, (metric, label) in zip(axes, (("cpu_p95", "CPU"), ("memory_p95", "Memory"))):
        for k, runtime in enumerate(("thread", "process")):
            for i, c in enumerate(CONTROLLER_ORDER):
                style = _bar_style(c)
                style["hatch"] = "...." if runtime == "process" else style["hatch"]
                ax.bar(x[i] + (k - 0.5) * width, means.loc[(runtime, c), metric], width=width * 0.92, **style)
        ax.set_xticks(x)
        ax.set_xticklabels([CONTROLLER_LABEL[c] for c in CONTROLLER_ORDER], rotation=32, ha="right")
        ax.set_title(f"{label} 95th percentile")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Cluster utilization (%)")
    handles = [
        Patch(facecolor="#9aa1ab", edgecolor="white", label="Thread runtime (left bar)"),
        Patch(facecolor="#9aa1ab", edgecolor="white", hatch="....", label="Process runtime (right bar, dotted)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "controlled_resource_pressure")


def _action_panel(ax: plt.Axes, shares: pd.DataFrame, entities: list[str]) -> None:
    left = np.zeros(len(entities))
    for action in ACTION_ORDER:
        values = shares.reindex(entities)[action].fillna(0.0).to_numpy()
        ax.barh(range(len(entities)), values, left=left, color=ACTION_COLOR[action], edgecolor="white",
                linewidth=0.5, height=0.7)
        left += values
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(entities)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def controlled_action_frequency() -> None:
    """Share of frozen-evaluation decisions per action, pooled over profiles and evaluation seeds."""
    rows = _read(CONTROLLED, "controlled_action_counts")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), sharey=True)
    for ax, runtime in zip(axes, ("thread", "process")):
        counts = rows[rows.runtime == runtime].groupby(["controller", "action"])["count"].sum().unstack(fill_value=0)
        shares = counts.div(counts.sum(axis=1), axis=0)
        _action_panel(ax, shares, CONTROLLER_ORDER)
        ax.set_yticklabels([CONTROLLER_LABEL[c] for c in CONTROLLER_ORDER])
        ax.set_title(RUNTIME_LABEL[runtime])
        ax.set_xlabel("Share of decisions")
    axes[0].invert_yaxis()  # shared y axis: invert once so the first controller is on top
    handles = [Patch(color=ACTION_COLOR[a], label=ACTION_LABEL[a]) for a in ACTION_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=2.5)
    _save(fig, "controlled_action_frequency")


def coverage_violations_by_profile() -> None:
    """Coverage violation events by profile, split by bound direction, for both evaluation stages."""
    controlled = _read(CONTROLLED, "controlled_violation_bounds")
    live = _read(LIVE, "live_violation_bounds")
    controlled = controlled[controlled.metric == "coverage"]
    live = live[live.metric == "coverage"]
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 4.9))
    panels = (
        (axes[0], controlled, "controller", CONTROLLER_ORDER, "Controlled stage: both runtimes, five evaluation seeds"),
        (axes[1], live, "variant", ["static", "dqn", "ppo"], "Live stage: both regimes, five live seeds"),
    )
    for ax, data, key, entities, title in panels:
        width = 0.8 / len(entities)
        for j, entity in enumerate(entities):
            sub = data[data[key] == entity]
            above = sub[sub.bound == "maximum"].groupby("profile").events.sum().reindex(PROFILE_ORDER, fill_value=0)
            below = sub[sub.bound == "minimum"].groupby("profile").events.sum().reindex(PROFILE_ORDER, fill_value=0)
            if below.sum() > 0:
                raise RuntimeError("Lower-bound coverage events present; this figure assumes none")
            xs = [i + (j - (len(entities) - 1) / 2) * width for i in range(len(PROFILE_ORDER))]
            ax.bar(xs, above.to_numpy(), width=width * 0.92, **_bar_style(entity))
        ax.set_xticks(range(len(PROFILE_ORDER)))
        ax.set_xticklabels([PROFILE_LABEL[p] for p in PROFILE_ORDER])
        total_below = int(data[data.bound == "minimum"].events.sum())
        ax.set_title(f"{title} (events below contract minimum: {total_below})")
        ax.set_ylabel("Events above maximum")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.legend(handles=_controller_handles(CONTROLLER_ORDER), loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, frameon=False, columnspacing=1.1, handlelength=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2)
    _save(fig, "coverage_violations_by_profile")


def live_paired_fidelity_delta() -> None:
    """Paired spatial-fidelity delta (frozen learned minus static midpoint) per live seed."""
    rows = _read(LIVE, "live_paired_deltas")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True)
    for ax, scenario in zip(axes, ("realistic", "concurrency")):
        for j, variant in enumerate(("dqn", "ppo")):
            sub = rows[(rows.scenario == scenario) & (rows.variant == variant)].sort_values("live_seed")
            center, half = _t_interval(sub.spatial_fidelity_mean_delta)
            xs = np.arange(len(sub)) + (j - 0.5) * 0.22
            ax.scatter(xs, sub.spatial_fidelity_mean_delta, marker=CONTROLLER_MARKER[variant],
                       color=CONTROLLER_COLOR[variant], s=26, zorder=3, label=CONTROLLER_LABEL[variant])
            ax.errorbar(len(sub) + 0.4 + j * 0.35, center, yerr=half, fmt=CONTROLLER_MARKER[variant],
                        color=CONTROLLER_COLOR[variant], ecolor="black", capsize=3, markersize=5, linewidth=0.9)
        seeds = sorted(rows.live_seed.unique())
        ax.set_xticks(list(range(len(seeds))) + [len(seeds) + 0.575])
        ax.set_xticklabels([str(s) for s in seeds] + ["Mean\n(95% CI)"])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(SCENARIO_LABEL[scenario])
        ax.set_xlabel("Live seed")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Spatial fidelity delta vs. Static")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "live_paired_fidelity_delta")


def live_admission_outcomes() -> None:
    """Requests submitted, admitted on arrival, queued on arrival, and evaluated, summed over five trials."""
    rows = _read(LIVE, "live_runs")
    totals = rows.groupby(["scenario", "variant"])[
        ["submitted", "initially_accepted", "initially_queued", "evaluated_requests"]
    ].sum()
    measures = [
        ("submitted", "Submitted"),
        ("initially_accepted", "Admitted\non arrival"),
        ("initially_queued", "Queued\non arrival"),
        ("evaluated_requests", "Evaluated"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    variants = ["static", "dqn", "ppo"]
    width = 0.26
    for ax, scenario in zip(axes, ("realistic", "concurrency")):
        for j, variant in enumerate(variants):
            values = [totals.loc[(scenario, variant), m] for m, _ in measures]
            xs = np.arange(len(measures)) + (j - 1) * width
            bars = ax.bar(xs, values, width=width * 0.92, **_bar_style(variant))
            ax.bar_label(bars, labels=[str(int(v)) for v in values], fontsize=6.5, padding=1.5)
        ax.set_xticks(range(len(measures)))
        ax.set_xticklabels([label for _, label in measures], fontsize=8)
        ax.set_title(SCENARIO_LABEL[scenario])
        ax.margins(y=0.12)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Requests (five trials)")
    fig.legend(handles=_controller_handles(variants), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "live_admission_outcomes")


def live_cumulative_reward() -> None:
    """Cumulative decision reward over trial time, averaged over the five paired live seeds."""
    rows = _read(LIVE, "live_reward_timeline")
    mean = rows.groupby(["scenario", "variant", "minute"], as_index=False).cumulative_reward.mean()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    for ax, scenario in zip(axes, ("realistic", "concurrency")):
        for variant in ("static", "dqn", "ppo"):
            sub = mean[(mean.scenario == scenario) & (mean.variant == variant)]
            ax.plot(sub.minute, sub.cumulative_reward, color=CONTROLLER_COLOR[variant], linewidth=1.5,
                    marker=CONTROLLER_MARKER[variant], markevery=10, markersize=4, label=CONTROLLER_LABEL[variant])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlim(0, 30)
        ax.set_title(SCENARIO_LABEL[scenario])
        ax.set_xlabel("Elapsed time (minutes)")
        ax.grid(alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Mean cumulative reward")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "live_cumulative_reward")


MAIN = [controlled_mean_reward, controlled_reward_by_profile, live_request_load, live_reward_by_seed]
SUPPLEMENTARY = [
    controlled_paired_reward_deltas,
    controlled_fidelity_vs_duty,
    controlled_resource_pressure,
    controlled_action_frequency,
    coverage_violations_by_profile,
    live_paired_fidelity_delta,
    live_admission_outcomes,
    live_cumulative_reward,
]
EXPECTED_MAIN = [
    "controlled_mean_reward_thread",
    "controlled_mean_reward_process",
    "controlled_reward_by_profile_thread",
    "controlled_reward_by_profile_process",
    "live_request_load_realistic",
    "live_request_load_concurrency",
    "live_reward_by_seed_realistic",
    "live_reward_by_seed_concurrency",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--main", action="store_true", help="Generate only the eight main panels")
    args = parser.parse_args()
    for figure in MAIN + ([] if args.main else SUPPLEMENTARY):
        figure()
    if args.main:
        expected = EXPECTED_MAIN
    else:
        expected = EXPECTED_MAIN + [f.__name__ for f in SUPPLEMENTARY]
    missing = [
        f"{fmt}/{name}.{fmt}"
        for name in expected
        for fmt in ("pdf", "png")
        if not (OUT / fmt / f"{name}.{fmt}").is_file() or (OUT / fmt / f"{name}.{fmt}").stat().st_size == 0
    ]
    if missing:
        sys.exit(f"Figure generation incomplete: {missing}")
    print(f"Generated {len(FIGURES)} figures (PDF and PNG) in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
