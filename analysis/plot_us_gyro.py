"""Check one trial: thigh gyro, ultrasonic distance and echo/timeout on one figure.

Strides are marked at each minimum of the thigh gyro gy (one per stride, i.e.
every 2 steps), so you can see whether the ultrasonic changes once or twice per
stride. Echoes closer than CLOSE_MM are drawn as "other thigh", the rest as
"past the thigh"; the distance panel is log-scaled so both show.

Usage:
    python analysis/plot_us_gyro.py data/raw/bench_01.csv
    python analysis/plot_us_gyro.py data/raw/bench_01.csv --out docs/figures/bench_01_us_gyro.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

CLOSE_MM = 100
STRIDE_PROMINENCE = 60      # deg/s, gy minima
COLOR_CLOSE = "#eb6834"     # categorical slot 2
COLOR_FAR = "#2a78d6"       # categorical slot 1
INK = "#1d1d1b"
MISS = "#b9b7b1"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SHADE = "#f0efec"


def style(ax, unit):
    ax.set_ylabel(unit, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description="Plot gyro, ultrasonic distance and echo/timeout for a trial.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, help="save to this image file instead of showing a window")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t = ((df["t_ms"] - df["t_ms"].iloc[0]) / 1000).to_numpy()
    meta_path = args.csv.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    gy = df["gy"].to_numpy()
    strides, _ = find_peaks(-gy, prominence=STRIDE_PROMINENCE, distance=20)
    hit = df["us_hit"].to_numpy() == 1
    close = hit & (df["us_mm"].to_numpy() < CLOSE_MM)
    far = hit & ~close

    fig, (a1, a2, a3) = plt.subplots(3, 1, sharex=True, figsize=(12, 7.5),
                                     gridspec_kw={"height_ratios": [3, 3, 1]})
    for ax in (a1, a2, a3):
        for s in t[strides]:
            ax.axvline(s, color=GRID, linewidth=1.0, zorder=0)

    a1.plot(t, gy, color=INK, linewidth=1.2)
    a1.scatter(t[strides], gy[strides], s=30, color=INK, zorder=3)
    a1.axhline(0, color=GRID, linewidth=0.8)
    style(a1, "thigh gyro gy (deg/s)")
    a1.set_title(f"Thigh swing: {len(strides)} strides (gy minima, gray lines) = {2 * len(strides)} steps",
                 loc="left", color=MUTED, fontsize=10)

    a2.scatter(t[close], df["us_mm"][close], s=14, color=COLOR_CLOSE, label=f"echo < {CLOSE_MM} mm (other thigh)",
               zorder=3)
    a2.scatter(t[far], df["us_mm"][far], s=14, color=COLOR_FAR, label=f"echo ≥ {CLOSE_MM} mm (past the thigh)",
               zorder=3)
    a2.scatter(t[~hit], np.full((~hit).sum(), 1000), s=10, marker="|", color=MISS, label="timeout (1000)")
    a2.set_yscale("log")
    a2.set_ylim(15, 1400)
    a2.set_yticks([20, 50, 100, 200, 500, 1000])
    a2.set_yticklabels(["20", "50", "100", "200", "500", "1000"])
    a2.legend(loc="upper left", bbox_to_anchor=(0, 1.16), frameon=False, fontsize=8.5, labelcolor=MUTED, ncols=3)
    style(a2, "ultrasonic (mm, log)")

    a3.fill_between(t, 0, hit.astype(float), step="mid", color=INK, linewidth=0)
    a3.set_ylim(0, 1.05)
    a3.set_yticks([0, 1])
    style(a3, "us_hit")
    a3.set_xlabel("time (s)", color=MUTED)

    bits = [f"{meta['counted_steps']} counted steps" if meta.get("counted_steps") is not None else None,
            meta.get("speed"), meta.get("notes")]
    fig.suptitle(f"{args.csv.stem}: " + ", ".join(b for b in bits if b), color=TEXT, x=0.01, ha="left",
                 fontsize=12)
    fig.text(0.01, 0.94, f"{len(df)} samples, {t[-1]:.1f} s, echo {100 * hit.mean():.0f}% "
             f"(close {close.sum()}, far {far.sum()})", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.8)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
