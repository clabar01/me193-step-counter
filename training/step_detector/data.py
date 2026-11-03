"""Load trial recordings and turn them into model inputs.

A trial is the board's <session>_<trial>.csv + .json (see the README). Every
trial goes through the same steps, in training and later live:

  1. despike() the IMU: replace single-sample I2C glitches.
  2. Line the ultrasonic up with the IMU: the sketch fires each ping right
     after an IMU read and logs its result on the NEXT row
     (json us_lag_samples, 1), so row i's ping is moved back to row i - 1.
  3. Clip the ultrasonic at US_CLIP_MM (README, "For training"): the other
     thigh reads 16-100 mm, anything farther is walls and furniture, which
     depend on the room. Farther echoes and timeouts both become
     us_mm = US_CLIP_MM, us_hit = 0, so the model can't learn the room.

bench_* trials are hardware checks and are always skipped.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

AXES = ("ax", "ay", "az", "gx", "gy", "gz")
CHANNELS = AXES + ("us_mm", "us_hit")       # the model's 8 input channels, in this order
US_CLIP_MM = 150
FS = 50

# A sample is a spike if, on any axis, it is more than SPIKE from BOTH
# neighbours while the neighbours are within AGREE of each other. Tuned on the
# first real recordings (2026-10-23, gait project): the one corrupted I2C read
# jumped 11 g and 750 deg/s for a single sample on four axes, while real
# single-axis heel-strike peaks of 4-5.5 g (gyro smooth) must survive.
SPIKE = np.array([6.0, 6.0, 6.0, 500.0, 500.0, 500.0])
AGREE = SPIKE / 2


@dataclass
class Trial:
    name: str
    session: str
    meta: dict
    x: np.ndarray                # (N, 8) float32 in CHANNELS order, preprocessed, NOT normalized
    t_ms: np.ndarray
    despiked: int = 0
    truth: dict | None = field(default=None)    # simulator only: true step times

    @property
    def counted(self):
        return self.meta.get("counted_steps")

    @property
    def gy(self):
        return self.x[:, CHANNELS.index("gy")]


def despike(samples):
    """Replace single-sample glitches with the average of their neighbours.

    samples: (N, 6) array-like in AXES order. Returns (cleaned copy, indices
    of the replaced samples). A corrupted read garbles every axis of that
    sample, so the whole sample is replaced, not just the axis that tripped
    the check. The first and last samples have only one neighbour and are
    left as they are; two bad samples in a row are not caught.
    """
    x = np.array(samples, dtype=float)
    if len(x) < 3:
        return x, np.array([], dtype=int)

    prev, cur, nxt = x[:-2], x[1:-1], x[2:]
    far = (np.abs(cur - prev) > SPIKE) & (np.abs(cur - nxt) > SPIKE)
    agree = np.abs(prev - nxt) <= AGREE
    bad = np.flatnonzero((far & agree).any(axis=1)) + 1

    # Neighbours of a bad sample are never bad themselves here (they'd need
    # their own neighbours to agree across the spike), so this is safe in one pass.
    x[bad] = (x[bad - 1] + x[bad + 1]) / 2
    return x, bad


def align_ultrasonic(values, lag):
    """Move each ping back `lag` rows to the IMU sample it was fired after.
    The last `lag` rows have no ping of their own and repeat the last one."""
    values = np.asarray(values)
    if lag <= 0 or len(values) <= lag:
        return values.copy()
    return np.concatenate([values[lag:], np.repeat(values[-1:], lag)])


def clip_ultrasonic(us_mm, us_hit):
    """Keep only near echoes (< US_CLIP_MM); everything else reads as no echo."""
    near = (np.asarray(us_hit) == 1) & (np.asarray(us_mm) < US_CLIP_MM)
    return np.where(near, us_mm, US_CLIP_MM).astype(float), near.astype(float)


def preprocess(df, lag):
    """Board CSV rows -> (N, 8) inputs in CHANNELS order, and how many samples were despiked."""
    imu, bad = despike(df[list(AXES)].to_numpy(float))
    us_mm = align_ultrasonic(df["us_mm"].to_numpy(), lag)
    us_hit = align_ultrasonic(df["us_hit"].to_numpy(), lag)
    us_mm, us_hit = clip_ultrasonic(us_mm, us_hit)
    return np.column_stack([imu, us_mm, us_hit]).astype(np.float32), len(bad)


def load_trials(data_dir, truth_dir=None):
    """Every <name>.json + <name>.csv pair in data_dir, except bench_* trials."""
    data_dir = Path(data_dir)
    trials, skipped = [], []
    for meta_path in sorted(data_dir.glob("*.json")):
        name = meta_path.stem
        if name.startswith("bench_"):
            skipped.append(name)
            continue
        meta = json.loads(meta_path.read_text())
        df = pd.read_csv(data_dir / meta.get("csv", f"{name}.csv"))
        x, n_bad = preprocess(df, int(meta.get("us_lag_samples", 1)))
        truth = None
        if truth_dir and (Path(truth_dir) / f"{name}.json").exists():
            truth = json.loads((Path(truth_dir) / f"{name}.json").read_text())
        trials.append(Trial(name, meta.get("session", name.split("_")[0]), meta, x,
                            df["t_ms"].to_numpy(), n_bad, truth))
    return trials, skipped


def true_step_samples(trial):
    """Simulator truth: step times as sample indices into this trial (or None)."""
    if not trial.truth:
        return None
    return np.round((np.array(trial.truth["step_t_ms"]) - trial.t_ms[0]) / (1000 / FS)).astype(int)
