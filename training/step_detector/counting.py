"""Turning the CNN output into a step count, and scoring step timing.

Three counts:
  - peaks: the output's 100 ms moving sum (one step's mass, see losses.py),
    peaks above PEAK_HEIGHT at least PEAK_GAP_S apart. The model places a step
    to within a sample or two, not on one exact sample, so its mass is spread
    over ~3-5 samples and single-sample values are often below 0.5; their
    100 ms sum is ~1. train.py picks its kept epoch with this count.
  - count_steps (THE counting rule, used by analysis/evaluate.py and the board):
    peaks, plus an edge step at the start / end of each walking bout when the
    gyro stride detector has a step there that the CNN missed. The CNN nearly
    always misses the last step of a walk (its output there is ~0, not a weak
    peak). Chosen over a lower threshold and round(sum) by cross-validation
    inside the training trials (analysis/counting_rule.py).
  - sum: the plain sum of the output, which is what the count loss trains.
"""

import numpy as np
from scipy.signal import find_peaks

from strides import step_times

FS = 50
STEP_SAMPLES = 5            # 100 ms, as in losses.py
PEAK_HEIGHT = 0.5
PEAK_GAP_S = 0.25           # two steps can't be closer than this (jogging is ~0.38 s)
MATCH_S = 0.10              # timing tolerance for event F1
EDGE_GAP_S = 1.5            # no step for this long ends a walking bout (= the board's STEP_TIMEOUT_S)
# The gyro check only looks at gy in a window around the bout's edge step, in
# samples, so the board (which keeps a few seconds of samples) computes the
# same thing. The start of a bout is checked once its 2nd step is known (or at
# its end, for a 1-step bout), the end once the bout is over.
START_WINDOW = (-150, 70)
END_WINDOW = (-150, 100)
END_GAP_STEPS = 6           # the end check's step interval: median of the bout's last 6 intervals
LONE_GAP = 30               # the step interval assumed for a 1-step bout


def step_mass(y):
    """100 ms moving sum of the output."""
    return np.convolve(np.asarray(y, dtype=float), np.ones(STEP_SAMPLES), "same")


def peak_steps(y):
    """Sample indices of the counted steps."""
    peaks, _ = find_peaks(step_mass(y), height=PEAK_HEIGHT, distance=int(PEAK_GAP_S * FS))
    return peaks


def bouts(steps, gap):
    """Split step indices into walking bouts: a gap of >= gap samples starts a new bout."""
    out = []
    for s in steps:
        if out and s - out[-1][-1] < gap:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def edge_step(gy, step, gap, side):
    """The step the gyro detector has 0.5-1.5 step intervals before (side="start") or after
    (side="end") `step`, closest to one interval away, using only gy in that side's window;
    or None."""
    lo, hi = START_WINDOW if side == "start" else END_WINDOW
    a, b = max(0, step + lo), min(len(gy), step + hi)
    if b - a < 3:
        return None
    sign = -1 if side == "start" else 1
    cands = [int(g) + a for g in step_times(gy[a:b]) if 0.5 * gap <= sign * (int(g) + a - step) <= 1.5 * gap]
    return min(cands, key=lambda g: abs(sign * (g - step) - gap)) if cands else None


def bout_gaps(b):
    """(start check's interval, end check's interval) for one bout of step indices."""
    start = b[1] - b[0] if len(b) > 1 else LONE_GAP
    end = float(np.median(np.diff(b[-(END_GAP_STEPS + 1):]))) if len(b) > 1 else LONE_GAP
    return start, end


def edge_steps(peaks, gy):
    """The edge steps count_steps adds to the CNN's peaks. Bouts are checked in order, start
    then end; an edge step closer than PEAK_GAP_S to a step already known (this and earlier
    bouts' peaks, edge steps added so far) is the same step and isn't added again (a missed
    step mid-walk can split a walk into two bouts that both find it)."""
    out, known = [], []
    for b in bouts(list(peaks), int(EDGE_GAP_S * FS)):
        known += b
        g_start, g_end = bout_gaps(b)
        for s in (edge_step(gy, b[0], g_start, "start"), edge_step(gy, b[-1], g_end, "end")):
            if s is not None and all(abs(s - k) >= PEAK_GAP_S * FS for k in known):
                out.append(s)
                known.append(s)
    return np.array(sorted(out), dtype=int)


def count_steps(y, gy):
    """Step indices by THE counting rule: CNN peaks plus gyro-checked edge steps."""
    p = peak_steps(y)
    return np.sort(np.concatenate([p, edge_steps(p, gy)])).astype(int)


def sum_count(y):
    """The plain sum of the output (what the count loss trains)."""
    return float(np.sum(y))


def event_f1(pred, true, tol_s=MATCH_S):
    """Precision, recall and F1 of predicted step times vs true ones.

    Each true step can be matched by at most one predicted step within
    +-tol_s, closest pairs first. Returns (precision, recall, f1).
    """
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    if len(pred) == 0 and len(true) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0 or len(true) == 0:
        return (0.0 if len(pred) else 1.0), (0.0 if len(true) else 1.0), 0.0
    tol = tol_s * FS
    d = np.abs(pred[:, None] - true[None, :])
    pairs = sorted(zip(*np.nonzero(d <= tol)), key=lambda ij: d[ij])
    used_p, used_t = set(), set()
    for i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    tp = len(used_p)
    precision, recall = tp / len(pred), tp / len(true)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return precision, recall, f1
