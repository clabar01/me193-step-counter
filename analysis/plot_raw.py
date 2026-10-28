"""Plot accel and gyro magnitude over time for one recording CSV.

Usage:
    python analysis/plot_raw.py data/raw/ceci_walking_20261023-141500.csv
    python analysis/plot_raw.py data/raw/ceci_walking_20261023-141500.csv --out analysis/figures/walking_raw.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LINE_COLOR = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def main():
    ap = argparse.ArgumentParser(description="Plot |accel| and |gyro| for a recording CSV.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, help="save to this image file instead of showing a window")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t = (df["t"] - df["t"].iloc[0]) / 1000
    acc = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    gyr = np.sqrt(df["gx"] ** 2 + df["gy"] ** 2 + df["gz"] ** 2)

    # Two panels, not a twin axis: g and deg/s are different units.
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5.5))
    for ax, y, unit in ((ax1, acc, "|accel| (g)"), (ax2, gyr, "|gyro| (deg/s)")):
        ax.plot(t, y, color=LINE_COLOR, linewidth=1.2)
        ax.set_ylabel(unit, color=MUTED)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    ax2.set_xlabel("time (s)", color=MUTED)

    label = df["label"].iloc[0] if "label" in df else "?"
    subject = df["subject"].iloc[0] if "subject" in df else "?"
    fig.suptitle(f"{label}, subject {subject}", color=TEXT, x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.92, f"{args.csv.name}   {len(df)} samples, {t.iloc[-1]:.1f} s",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.5)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
