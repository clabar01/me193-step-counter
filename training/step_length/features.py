"""Split a walking trial into steps and compute per-step features.

Uses the same loading and preprocessing as the step detector
(../step_detector/data.py: BODY_R frame from the board, despike, ultrasonic
lined up and clipped at 150 mm, bench_* skipped) and the same gyro stride
detector (../step_detector/strides.py) to find the steps.

Only the main walking bout is used: the longest run of strides with no gap
over BOUT_GAP stride periods. A leg movement after you stop (e.g. B_05, ~5 s
after the last step) is not part of the measured distance.

Per step k (from step k's heel strike to the next one):
  step_time_s   time to the next step
  swing_deg     thigh swing angle: gy integrated over one stride centred on
                the step, minus a straight line (drift removal: the thigh ends
                a stride where it started), max - min
  acc_range_g   |accel| max - min within the step (the Weinberg feature)
  us_min_mm     closest ultrasonic reading within the step (150 = none):
                how close the thighs come as they pass
  us_burst_ms   longest run of close echoes within the step: how long the
                thighs are side by side
Each step holds one thigh pass (passes are ~a quarter stride after each heel
strike), so the ultrasonic features describe that pass.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "step_detector"))   # after this folder: both have plots.py
from data import CHANNELS, US_CLIP_MM, load_trials  # noqa: E402
from strides import step_times, swing_peaks  # noqa: E402

FS = 50
BOUT_GAP = 1.8              # strides further apart than this x the median are separate bouts
FEATURES = ("step_time_s", "swing_deg", "acc_range_g", "us_min_mm", "us_burst_ms")
IX = {c: i for i, c in enumerate(CHANNELS)}


def walking_bout(peaks):
    """The longest run of swing peaks without a long gap (sample indices)."""
    if len(peaks) < 2:
        return peaks
    gaps = np.diff(peaks)
    breaks = np.flatnonzero(gaps > BOUT_GAP * np.median(gaps))
    runs = np.split(peaks, breaks + 1)
    return max(runs, key=len)


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def step_features(x, steps=None):
    """(steps, n_features) array for one trial's preprocessed (N, 8) inputs, plus step indices.

    steps: sample indices of the steps to describe (e.g. the CNN's, as on the
    board); by default the gyro stride detector's steps in the main walking bout."""
    gy = x[:, IX["gy"]].astype(float)
    if steps is None:
        peaks, period = swing_peaks(gy)
        bout = walking_bout(peaks)
        if len(bout) < 2:
            return np.zeros((0, len(FEATURES))), np.array([], dtype=int)
        stride = int(np.median(np.diff(bout)))
        lo, hi = bout[0] - stride, bout[-1] + stride        # half-steps at the ends belong to the bout
        steps = np.array([s for s in step_times(gy) if lo <= s <= hi])
    else:
        steps = np.asarray(steps, dtype=int)
        if len(steps) < 2:
            return np.zeros((0, len(FEATURES))), steps
        stride = int(2 * np.median(np.diff(steps)))
    acc = np.linalg.norm(x[:, :3], axis=1)
    us, near = x[:, IX["us_mm"]], x[:, IX["us_hit"]] == 1

    rows = []
    for k, s in enumerate(steps):
        nxt = steps[k + 1] if k + 1 < len(steps) else s + (s - steps[k - 1] if k else stride // 2)
        nxt = min(nxt, len(x) - 1)
        a, b = max(0, s - stride // 2), min(len(x), s + stride // 2)
        angle = np.cumsum(gy[a:b]) / FS
        angle -= np.linspace(angle[0], angle[-1], len(angle))     # remove drift over the stride
        within = slice(s, max(nxt, s + 1))
        rows.append([
            (nxt - s) / FS,
            angle.max() - angle.min(),
            acc[within].max() - acc[within].min(),
            us[within][near[within]].min() if near[within].any() else US_CLIP_MM,
            longest_run(near[within]) * 1000 / FS,
        ])
    return np.array(rows), steps


def distance_trials(data_dir):
    """Every distance trial with a known distance: (trial, features, step indices)."""
    trials, skipped = load_trials(data_dir)
    out = []
    for tr in trials:
        if tr.meta.get("type") == "distance" and tr.meta.get("distance_m"):
            f, steps = step_features(tr.x)
            out.append((tr, f, steps))
    return out, skipped
