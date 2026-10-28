"""Plot one trial recording: thigh accel along x, thigh gyro gy, ultrasonic distance.

Reads <name>.csv and, if it is next to it, <name>.json for the title.

Usage:
    python analysis/plot_raw.py data/raw/20261028_01.csv
    python analysis/plot_raw.py data/raw/20261028_01.csv --out analysis/figures/20261028_01.png
    python analysis/plot_raw.py data/raw/20261028_01.csv --seconds 10      # first 10 s only
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LINE_COLOR = "#2a78d6"
MISS_COLOR = "#b9b7b1"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def style(ax, unit):
    ax.set_ylabel(unit, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description="Plot accel x, gyro y and ultrasonic distance for a trial CSV.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, help="save to this image file instead of showing a window")
    ap.add_argument("--seconds", type=float, help="only plot the first this many seconds")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000
    if args.seconds:
        keep = t <= args.seconds
        df, t = df[keep], t[keep]
    meta_path = args.csv.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Separate panels, not twin axes: g, deg/s and mm are different units.
    fig, (a1, a2, a3) = plt.subplots(3, 1, sharex=True, figsize=(11, 7))
    a1.plot(t, df["ax"], color=LINE_COLOR, linewidth=1.0)
    style(a1, "ax (g)")
    a2.plot(t, df["gy"], color=LINE_COLOR, linewidth=1.0)
    a2.axhline(0, color=GRID, linewidth=0.8)
    style(a2, "gy (deg/s)")
    hit = df["us_hit"] == 1
    a3.scatter(t[hit], df["us_mm"][hit], s=6, color=LINE_COLOR, label="echo")
    a3.scatter(t[~hit], df["us_mm"][~hit], s=6, marker="|", color=MISS_COLOR, label="timeout (1000)")
    a3.set_ylim(0, 1150)
    a3.legend(loc="upper right", bbox_to_anchor=(1, 1.08), frameon=False, fontsize=8,
              labelcolor=MUTED, ncols=2)
    style(a3, "ultrasonic (mm)")
    a3.set_xlabel("time (s)", color=MUTED)

    if meta:
        bits = [meta.get("type"), meta.get("speed")]
        if meta.get("counted_steps") is not None:
            bits.append(f"{meta['counted_steps']} counted steps")
        if meta.get("distance_m"):
            bits.append(f"{meta['distance_m']:g} m")
        title = f"{args.csv.stem}: " + ", ".join(str(b) for b in bits if b)
    else:
        title = args.csv.stem
    fig.suptitle(title, color=TEXT, x=0.01, ha="left", fontsize=13)
    source = f"   source: {meta['source']}" if meta.get("source") else ""
    fig.text(0.01, 0.935, f"{len(df)} samples, {t.iloc[-1]:.1f} s, echo {100 * hit.mean():.0f}%{source}",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.5)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
