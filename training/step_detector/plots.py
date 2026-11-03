"""Figures for the step detector: loss curves and per-trial outputs."""

import matplotlib.pyplot as plt
import numpy as np

from counting import PEAK_HEIGHT, step_mass
from data import CHANNELS, US_CLIP_MM

FS = 50
INK = "#1d1d1b"
SERIES_1 = "#2a78d6"        # categorical slot 1
SERIES_2 = "#eb6834"        # categorical slot 2
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def style(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def loss_curves(history, path, title):
    """Warm start (pseudo-label loss) and fine-tune (count loss, val count error)."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    pre = history["pretrain"]
    a1.plot(range(1, len(pre) + 1), pre, color=INK, linewidth=2)
    a1.set_title("Warm start: pseudo-label loss", loc="left", color=TEXT, fontsize=11)
    style(a1, "weighted BCE")
    a1.set_xlabel("epoch", color=MUTED)

    ft = history["finetune"]
    ep = np.arange(1, len(ft["train_count"]) + 1)
    a2.plot(ep, ft["train_count"], color=INK, linewidth=2)
    a2.set_yscale("log")
    a2.set_title("Fine-tune: train count loss", loc="left", color=TEXT, fontsize=11)
    style(a2, "(sum - counted)$^2$, log")
    a2.set_xlabel("epoch", color=MUTED)

    a3.plot(ep, ft["val_peak_mae"], color=SERIES_1, linewidth=2, label="peaks")
    a3.plot(ep, ft["val_sum_mae"], color=SERIES_2, linewidth=2, label="sum")
    a3.axvline(history["best_epoch"], color=GRID, linewidth=1.2)
    a3.text(history["best_epoch"], a3.get_ylim()[1], " kept", color=MUTED, va="top", fontsize=9)
    a3.set_title("Fine-tune: validation count error", loc="left", color=TEXT, fontsize=11)
    style(a3, "mean |count - counted|, steps")
    a3.set_xlabel("epoch", color=MUTED)
    a3.legend(frameon=False, labelcolor=MUTED, fontsize=9)
    fig.suptitle(title, color=TEXT, x=0.01, ha="left", fontsize=12)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def trial_outputs(rows, path, title):
    """One block per trial: gy, clipped ultrasonic, CNN output with counted peaks.
    rows: dicts with trial, y (output), peaks, baseline (step samples), true (or None)."""
    fig = plt.figure(figsize=(12, 4.3 * len(rows)), layout="constrained")
    subs = np.atleast_1d(fig.subfigures(len(rows), 1, hspace=0.03))
    for sub, r in zip(subs, rows):
        trial = r["trial"]
        t = np.arange(len(trial.x)) / FS
        a1, a2, a3 = sub.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 2]})
        a1.plot(t, trial.gy, color=INK, linewidth=1)
        a1.scatter(t[r["baseline"]], np.full(len(r["baseline"]), trial.gy.max() * 1.08), marker="v", s=14,
                   color=MUTED, label=f"gyro detector steps ({len(r['baseline'])})")
        style(a1, "gy (deg/s)")
        a1.legend(loc="upper right", frameon=False, labelcolor=MUTED, fontsize=8.5)

        us = trial.x[:, CHANNELS.index("us_mm")]
        near = trial.x[:, CHANNELS.index("us_hit")] == 1
        a2.scatter(t[near], us[near], s=6, color=SERIES_2)
        a2.set_ylim(0, US_CLIP_MM + 10)
        style(a2, "us_mm")

        mass = step_mass(r["y"])
        a3.plot(t, mass, color=SERIES_1, linewidth=1.2)
        a3.scatter(t[r["peaks"]], mass[r["peaks"]], s=18, color=SERIES_1, zorder=3,
                   label=f"counted steps ({len(r['peaks'])})")
        if r["true"] is not None:
            for s in r["true"]:
                a3.axvline(s / FS, color=GRID, linewidth=1, zorder=0)
        a3.axhline(PEAK_HEIGHT, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        a3.set_ylim(-0.03, max(1.3, mass.max() * 1.05))
        style(a3, "step mass")
        a3.set_xlabel("time (s)" + ("; gray lines = true steps (simulator)" if r["true"] is not None else ""),
                      color=MUTED)
        a3.legend(loc="upper right", frameon=False, labelcolor=MUTED, fontsize=8.5)
        m = trial.meta
        sub.suptitle(f"{trial.name}: {m.get('speed')}, counted {trial.counted}, CNN {len(r['peaks'])} "
                     f"(sum {r['y'].sum():.1f}), gyro {len(r['baseline'])} - {m.get('notes') or ''}",
                     color=TEXT, x=0.01, ha="left", fontsize=11)
    fig.suptitle(title, color=MUTED, x=0.99, ha="right", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)
