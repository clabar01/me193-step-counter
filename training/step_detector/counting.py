"""Turning the CNN output into a step count, and scoring step timing.

Two counts:
  - peaks: the output's 100 ms moving sum (one step's mass, see losses.py),
    peaks above PEAK_HEIGHT at least PEAK_GAP_S apart. The model places a step
    to within a sample or two, not on one exact sample, so its mass is spread
    over ~3-5 samples and single-sample values are often below 0.5; their
    100 ms sum is ~1.
  - sum: the plain sum of the output, which is what the count loss trains.
"""

import numpy as np
from scipy.signal import find_peaks

FS = 50
STEP_SAMPLES = 5            # 100 ms, as in losses.py
PEAK_HEIGHT = 0.5
PEAK_GAP_S = 0.25           # two steps can't be closer than this (jogging is ~0.38 s)
MATCH_S = 0.10              # timing tolerance for event F1


def step_mass(y):
    """100 ms moving sum of the output."""
    return np.convolve(np.asarray(y, dtype=float), np.ones(STEP_SAMPLES), "same")


def peak_steps(y):
    """Sample indices of the counted steps."""
    peaks, _ = find_peaks(step_mass(y), height=PEAK_HEIGHT, distance=int(PEAK_GAP_S * FS))
    return peaks


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
