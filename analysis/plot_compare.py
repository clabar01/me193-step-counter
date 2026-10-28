"""Compare activities side by side: |accel| and |gyro| over the first 10 s.

One column per activity (standing, walking, running, air-biking). Top row is
accel magnitude, bottom row gyro magnitude, with the same y scale across each
row so the columns are directly comparable.

By default it uses the longest data/raw/<subject>_<activity>_*.csv recording
(by time span) for each activity.

Usage:
    python analysis/plot_compare.py
    python analysis/plot_compare.py --subject ceci --seconds 10 --no-show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "data" / "raw"
OUT = REPO / "analysis" / "figures" / "raw_comparison.png"
ACTIVITIES = ["standing", "walking", "running", "air-biking"]

LINE_COLOR = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def longest_recording(subject, activity):
    """Path and duration (s) of the longest recording, or (None, 0)."""
    best, best_secs = None, 0.0
    for path in sorted(RAW_DIR.glob(f"{subject}_{activity}_*.csv")):
        t = pd.read_csv(path, usecols=["t"])["t"]
        secs = (t.iloc[-1] - t.iloc[0]) / 1000 if len(t) else 0.0
        if secs > best_secs:
            best, best_secs = path, secs
    return best, best_secs


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description="Plot |accel| and |gyro| for each activity side by side.")
    ap.add_argument("--subject", default="ceci")
    ap.add_argument("--seconds", type=float, default=10.0, help="how much of each recording to show")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-show", action="store_true", help="only save the image")
    args = ap.parse_args()

    # Two rows, not a twin axis: g and deg/s are different units. sharey="row"
    # gives every column in a row the same scale.
    fig, axes = plt.subplots(2, len(ACTIVITIES), sharex=True, sharey="row",
                             figsize=(14, 6), squeeze=False)

    for col, activity in enumerate(ACTIVITIES):
        top, bottom = axes[0, col], axes[1, col]
        path, secs = longest_recording(args.subject, activity)
        if path is None:
            top.set_title(f"{activity}\n(no recording)", color=MUTED, fontsize=11)
            print(f"{activity}: no data/raw/{args.subject}_{activity}_*.csv")
            continue

        df = pd.read_csv(path)
        t = (df["t"] - df["t"].iloc[0]) / 1000
        keep = t <= args.seconds
        t = t[keep]
        acc = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)[keep]
        gyr = np.sqrt(df["gx"] ** 2 + df["gy"] ** 2 + df["gz"] ** 2)[keep]

        top.plot(t, acc, color=LINE_COLOR, linewidth=1.0)
        bottom.plot(t, gyr, color=LINE_COLOR, linewidth=1.0)
        top.set_title(activity, color=TEXT, fontsize=12, loc="left", pad=16)
        top.text(0, 1.02, f"{path.name}  ({secs:.1f} s)", transform=top.transAxes,
                 color=MUTED, fontsize=7, va="bottom")
        print(f"{activity}: {path.relative_to(REPO)} ({secs:.1f} s, showing {min(secs, args.seconds):.1f} s)")

    for ax in axes.flat:
        style(ax)
    axes[0, 0].set_ylabel("|accel| (g)", color=MUTED)
    axes[1, 0].set_ylabel("|gyro| (deg/s)", color=MUTED)
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylim(bottom=0)
    axes[0, 0].set_xlim(0, args.seconds)
    for ax in axes[1]:
        ax.set_xlabel("time (s)", color=MUTED)

    fig.suptitle(f"Raw IMU by activity, subject {args.subject}, first {args.seconds:g} s",
                 color=TEXT, x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
