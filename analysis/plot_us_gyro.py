"""Check trials: thigh gyro, ultrasonic distance and echo/timeout on one figure.

Strides are marked at each swing peak of the thigh gyro gy (the large positive
peak as this leg swings forward; one per stride, i.e. every 2 steps), found by
the same gyro stride detector the step-detector baseline uses
(training/step_detector/strides.py). Echoes closer than CLOSE_MM are drawn as "other thigh", the rest
as "past the thigh"; the distance panel is log-scaled so both show.

The thighs pass each other twice per stride: while this leg swings (near its
swing peak) and half a stride later while the other leg swings. Each close echo
burst while walking is matched to the nearest of those events; one more than a
quarter stride from both is "stray". Standing is where the gyro stays below
STILL_DPS for a whole second, reported separately before and after walking.

Give several CSVs to stack them (same time scale) and print stats for each.

Usage:
    python analysis/plot_us_gyro.py data/raw/bench_01.csv
    python analysis/plot_us_gyro.py data/raw/bench_01.csv data/raw/bench_04.csv --out docs/figures/bench_01_04_us_gyro.png
"""

import argparse
import json
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "step_detector"))
from strides import swing_peaks  # noqa: E402

CLOSE_MM = 100
MERGE_GAP = 2               # close echoes this many samples apart or fewer are one burst
STILL_DPS = 20              # gyro norm below this for 1 s = standing
SAMPLE_MS = 20
COLOR_CLOSE = "#eb6834"     # categorical slot 2
COLOR_FAR = "#2a78d6"       # categorical slot 1
INK = "#1d1d1b"
MISS = "#b9b7b1"
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


def bursts(mask):
    """(first, last) sample index of each run of True, merging gaps <= MERGE_GAP."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > MERGE_GAP + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, ends))


def standing(r, mask):
    n = mask.sum()
    if not n:
        return dict(s=0.0, hit=np.nan, close=np.nan, mm=np.nan)
    hits = r["us"][mask & r["hit"]]
    return dict(s=n * SAMPLE_MS / 1000, hit=r["hit"][mask].mean(), close=r["close"][mask].mean(),
                mm=np.median(hits) if len(hits) else np.nan)


def analyse(df):
    t = ((df["t_ms"] - df["t_ms"].iloc[0]) / 1000).to_numpy()
    gy = df["gy"].to_numpy()
    us = df["us_mm"].to_numpy()
    hit = df["us_hit"].to_numpy() == 1
    close = hit & (us < CLOSE_MM)

    gnorm = np.sqrt(df["gx"] ** 2 + df["gy"] ** 2 + df["gz"] ** 2)
    still = (gnorm.rolling(1000 // SAMPLE_MS, center=True, min_periods=1).max() < STILL_DPS).to_numpy()
    strides, period = swing_peaks(gy)
    s = t[strides]
    half = (period or 30) * SAMPLE_MS / 1000 / 2

    # this leg's pass at each swing peak, the other leg's half a stride later (between peaks)
    this_ev, other_ev = s, (s[:-1] + s[1:]) / 2
    walking = (t >= s[0] - half) & (t <= s[-1] + half) if len(s) else np.zeros(len(t), bool)
    at_this, at_other, stray, walk_ms = set(), set(), 0, []
    for a, b in bursts(close & walking):
        c = (t[a] + t[b]) / 2
        walk_ms.append((b - a + 1) * SAMPLE_MS)
        d_this = np.abs(this_ev - c)
        d_other = np.abs(other_ev - c) if len(other_ev) else np.array([np.inf])
        if min(d_this.min(), d_other.min()) > half / 2:
            stray += 1
        elif d_this.min() <= d_other.min():
            at_this.add(int(d_this.argmin()))
        else:
            at_other.add(int(d_other.argmin()))

    r = dict(t=t, gy=gy, us=us, hit=hit, close=close, far=hit & ~close, strides=strides,
             period_s=2 * half, n_strides=len(s), at_this=len(at_this), at_other=len(at_other),
             n_other=len(other_ev), stray=stray, burst_ms=np.array(walk_ms))
    first = s[0] if len(s) else t[-1]
    last = s[-1] if len(s) else t[-1]
    r["stand_before"] = standing(r, still & (t < first))
    r["stand_after"] = standing(r, still & (t > last))
    return r


def report(name, r):
    ms = r["burst_ms"]
    dur = f"median {np.median(ms):.0f} ms, range {ms.min():.0f}-{ms.max():.0f} ms" if len(ms) else "none"

    def stand(st):
        if not st["s"]:
            return "none"
        mm = "no echoes" if np.isnan(st["mm"]) else f"median {st['mm']:.0f} mm"
        return f"{st['s']:.1f} s, echo {100 * st['hit']:.0f}% (close {100 * st['close']:.0f}%), {mm}"

    print(f"{name}\n"
          f"  strides (swing peaks)          {r['n_strides']}  -> {2 * r['n_strides']} steps, "
          f"period {r['period_s']:.2f} s\n"
          f"  close burst, this leg swings   {r['at_this']} of {r['n_strides']}\n"
          f"  close burst, other leg swings  {r['at_other']} of {r['n_other']}\n"
          f"  stray close bursts             {r['stray']}\n"
          f"  burst length (walking)         {len(ms)} bursts, {dur}\n"
          f"  standing before walking        {stand(r['stand_before'])}\n"
          f"  standing after walking         {stand(r['stand_after'])}")


def plot(sub, name, r, meta, xmax):
    a1, a2, a3 = sub.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 3, 1]})
    t = r["t"]
    for ax in (a1, a2, a3):
        for x in t[r["strides"]]:
            ax.axvline(x, color=GRID, linewidth=1.0, zorder=0)
        ax.set_xlim(0, xmax)

    a1.plot(t, r["gy"], color=INK, linewidth=1.2)
    a1.scatter(t[r["strides"]], r["gy"][r["strides"]], s=30, color=INK, zorder=3)
    a1.axhline(0, color=GRID, linewidth=0.8)
    style(a1, "thigh gyro gy (deg/s)")
    a1.set_title(f"{r['n_strides']} strides (swing peaks, gray lines) = {2 * r['n_strides']} steps;  "
                 f"close echo as this leg swings {r['at_this']}/{r['n_strides']}, "
                 f"as the other leg swings {r['at_other']}/{r['n_other']}",
                 loc="left", color=MUTED, fontsize=9.5)

    close, far, hit = r["close"], r["far"], r["hit"]
    a2.scatter(t[close], r["us"][close], s=14, color=COLOR_CLOSE, label=f"echo < {CLOSE_MM} mm (other thigh)",
               zorder=3)
    a2.scatter(t[far], r["us"][far], s=14, color=COLOR_FAR, label=f"echo ≥ {CLOSE_MM} mm (past the thigh)",
               zorder=3)
    a2.scatter(t[~hit], np.full((~hit).sum(), 1000), s=10, marker="|", color=MISS, label="timeout (1000)")
    a2.set_yscale("log")
    a2.set_ylim(13, 1400)
    a2.set_yticks([20, 50, 100, 200, 500, 1000])
    a2.set_yticklabels(["20", "50", "100", "200", "500", "1000"])
    a2.legend(loc="upper left", bbox_to_anchor=(0, 1.18), frameon=False, fontsize=8.5, labelcolor=MUTED, ncols=3)
    style(a2, "ultrasonic (mm, log)")

    a3.fill_between(t, 0, hit.astype(float), step="mid", color=INK, linewidth=0)
    a3.set_ylim(0, 1.05)
    a3.set_yticks([0, 1])
    style(a3, "us_hit")
    a3.set_xlabel("time (s)", color=MUTED)

    bits = [f"{meta['counted_steps']} counted steps" if meta.get("counted_steps") is not None else None,
            meta.get("speed"), meta.get("notes")]
    sub.suptitle(f"{name}: " + ", ".join(b for b in bits if b), color=TEXT, x=0.01, ha="left", fontsize=12)


def main():
    ap = argparse.ArgumentParser(description="Plot gyro, ultrasonic distance and echo/timeout for trials.")
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, help="save to this image file instead of showing a window")
    args = ap.parse_args()

    trials = []
    for path in args.csv:
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        trials.append((path.stem, analyse(pd.read_csv(path)), meta))
    xmax = max(r["t"][-1] for _, r, _ in trials)

    fig = plt.figure(figsize=(12, 7.5 * len(trials)), layout="constrained")
    subs = np.atleast_1d(fig.subfigures(len(trials), 1, hspace=0.04))
    for sub, (name, r, meta) in zip(subs, trials):
        plot(sub, name, r, meta, xmax)
        report(name, r)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
