"""Fit the simulator's average stride from real knee-mount trials.

Reads bench_04 and A_01 from data/raw/ (the two knee-mount walking trials),
puts both in the current body frame (bench_04 was recorded with the old
axis-swap remap, so it is converted with the sketch's BODY_R), cuts them into
strides from one forward-swing gy peak to the next, averages the strides on a
0..1 phase grid and stores a Fourier series of that average per channel in
training/sim_template.json. training/simulator.py plays it back.

Only the average shape and a few timings are kept, not the recordings, so the
JSON can be committed. bench_* trials are used here only to shape the
simulator; they never go into training.

Usage:
    python scripts/fit_sim_template.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
OUT = REPO / "training" / "sim_template.json"
TRIALS = {"bench_04": "old", "A_01": "new"}    # which remap each was recorded with
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
BINS = 50
HARMONICS = 8

# board/sketch/sketch.ino: BODY_R (current) and the axis-swap remap it replaced
BODY_R = np.array([[-0.938, 0.078, -0.337],
                   [-0.104, -0.993, 0.060],
                   [-0.330, 0.091, 0.939]])
OLD_SIGN = np.array([-1.0, -1.0, 1.0])         # old: body = sign * sensor, same axis order


def to_current_frame(x, remap):
    """(N, 6) accel+gyro in the frame they were recorded in -> current body frame."""
    if remap == "new":
        return x
    acc, gyr = x[:, :3] * OLD_SIGN, x[:, 3:] * OLD_SIGN    # back to sensor axes
    return np.hstack([acc @ BODY_R.T, gyr @ BODY_R.T])


def strides(gy):
    """Forward-swing peaks: the big positive gy peak once per stride."""
    peaks, _ = find_peaks(gy, distance=40, height=80)
    return peaks


def main():
    templates, periods, hs_phase, pass_phase = [], [], [], []
    for name, remap in TRIALS.items():
        df = pd.read_csv(RAW / f"{name}.csv")
        x = to_current_frame(df[list(CHANNELS)].to_numpy(float), remap)
        us, hit = df["us_mm"].to_numpy(), df["us_hit"].to_numpy() == 1
        pk = strides(x[:, 4])
        grid = np.arange(BINS) / BINS
        for a, b in zip(pk[:-1], pk[1:]):
            phase = np.linspace(0, 1, b - a + 1)
            templates.append([np.interp(grid, phase, x[a:b + 1, c]) for c in range(6)])
            periods.append((b - a) * 0.02)
            gy = x[a:b, 4]
            down = np.flatnonzero((gy[:-1] > 0) & (gy[1:] <= 0))      # end of the forward swing
            if len(down):
                hs_phase.append(down[0] / (b - a))
            close = np.flatnonzero(hit[a:b] & (us[a:b] < 100)) / (b - a)
            pass_phase += close.tolist()
        print(f"{name}: {len(pk) - 1} strides")

    t = np.array(templates)                     # (strides, 6, BINS)
    mean = t.mean(axis=0)
    spread = t.std(axis=0).mean(axis=1)         # stride-to-stride variation per channel
    coef = np.fft.rfft(mean, axis=1) / BINS     # Fourier series of the average stride
    pp = np.array(pass_phase)
    doc = {
        "source": list(TRIALS),
        "note": "phase 0 = forward-swing gy peak; body frame of the current BODY_R remap",
        "channels": CHANNELS,
        "stride_s": round(float(np.median(periods)), 3),
        "harmonics": HARMONICS,
        "cos": np.round(2 * coef.real[:, :HARMONICS + 1], 4).tolist(),
        "sin": np.round(-2 * coef.imag[:, :HARMONICS + 1], 4).tolist(),
        "stride_spread": np.round(spread, 3).tolist(),
        "heel_strike_phase": round(float(np.mean(hs_phase)), 3),
        "thigh_pass_phase": [round(float(np.mean(pp[(pp > 0.25) & (pp < 0.7)])), 3),
                             round(float(np.mean(pp[pp >= 0.7])), 3)],
    }
    doc["cos"] = [[c[0] / 2] + c[1:] for c in doc["cos"]]    # the constant term isn't doubled
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(t)} strides, median {doc['stride_s']} s, heel strike at phase {doc['heel_strike_phase']}, "
          f"thigh passes at {doc['thigh_pass_phase']}\nsaved {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
