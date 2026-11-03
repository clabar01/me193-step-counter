"""Figures for the step-length model."""

import matplotlib.pyplot as plt
import numpy as np

from features import FEATURES

INK = "#1d1d1b"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")      # categorical slots 1-4 (validated)
UNITS = {"step_time_s": "step time (s)", "swing_deg": "thigh swing (deg)", "acc_range_g": "accel range (g)",
         "us_min_mm": "closest ultrasonic (mm)", "us_burst_ms": "ultrasonic burst (ms)"}


def style(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def features_vs_speed(data, path):
    """Every step's features against the trial's speed label (dots), median per speed (bar)."""
    speeds = [s for s in ("slow", "normal", "fast", "jog") if any(tr.meta.get("speed") == s for tr, _, _ in data)]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(3.1 * len(FEATURES), 3.6), layout="constrained")
    for ax, (j, name) in zip(axes, enumerate(FEATURES)):
        for i, sp in enumerate(speeds):
            vals = np.concatenate([f[:, j] for tr, f, _ in data if tr.meta.get("speed") == sp])
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=7, color=SERIES[0], alpha=0.35,
                       linewidths=0)
            med = np.median(vals)
            ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, linewidth=2.2)
        ax.set_xticks(range(len(speeds)), speeds)
        style(ax, UNITS[name])
    fig.suptitle("Per-step features by trial speed (dots = steps, line = median)", color=TEXT, x=0.01,
                 ha="left", fontsize=12)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def error_by_speed(df, methods, labels, speeds, path, title):
    """Grouped bars: mean |distance error| % per method, per speed and overall, value on each bar."""
    groups = [s for s in speeds if s in set(df["speed"])] + ["all"]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    for k, m in enumerate(methods):
        vals = []
        for g in groups:
            d = df if g == "all" else df[df["speed"] == g]
            vals.append(d[f"{m}_err%"].abs().mean())
        x = np.arange(len(groups)) + (k - (len(methods) - 1) / 2) * width
        ax.bar(x, vals, width * 0.92, color=SERIES[k], label=labels[m])
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    counts = [len(df) if g == "all" else int((df["speed"] == g).sum()) for g in groups]
    ax.set_xticks(range(len(groups)), [f"{g}\n({n} trial{'s' if n != 1 else ''})" for g, n in zip(groups, counts)])
    style(ax, "mean |distance error| (%)")
    ax.legend(frameon=False, labelcolor=MUTED, ncols=len(methods), loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    fig.savefig(path, dpi=120)
    plt.close(fig)
