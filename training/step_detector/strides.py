"""Gyro stride detector: the simple, hand-written way to count steps.

The thigh's forward swing is the big positive gy peak, once per stride (one
step of each leg), so steps = 2 x swing peaks. It is used three ways:
  - as the baseline the CNN has to beat (train.py),
  - to make pseudo-labels for the CNN's warm start (step times, below),
  - by analysis/plot_us_gyro.py to mark strides on the plots.

A peak counts as a swing if it is at least SWING_MIN_DPS high, at least 0.6
stride periods after the previous one (period from the gy autocorrelation, so
double bumps don't count twice) and at least half as high as the typical peak.
"""

import numpy as np
from scipy.signal import find_peaks

FS = 50
SWING_MIN_DPS = 60          # real swing peaks are 100-150 deg/s walking, ~280 jogging
MOVING_DPS = 20             # gy beyond this = moving (for the period estimate)
PERIOD_RANGE_S = (0.5, 2.5)


def stride_period(gy):
    """Stride period in samples from the gy autocorrelation while moving, or None."""
    moving = np.abs(gy) > MOVING_DPS
    if moving.sum() < 2 * FS:
        return None
    w = np.where(moving, gy, 0.0)
    w = w - w[moving].mean() * moving
    ac = np.correlate(w, w, "full")[len(w) - 1:]
    lo, hi = int(PERIOD_RANGE_S[0] * FS), min(int(PERIOD_RANGE_S[1] * FS), len(ac) - 1)
    if hi <= lo:
        return None
    return lo + int(np.argmax(ac[lo:hi]))


def swing_peaks(gy):
    """Sample indices of the forward-swing peaks (one per stride), and the period."""
    gy = np.asarray(gy, dtype=float)
    period = stride_period(gy)
    distance = int(0.6 * period) if period else int(0.4 * FS)
    peaks, props = find_peaks(gy, height=SWING_MIN_DPS, distance=distance)
    if len(peaks):
        peaks = peaks[props["peak_heights"] >= 0.5 * np.median(props["peak_heights"])]
    return peaks, period


def step_times(gy):
    """Sample indices of the steps the gyro detector implies (2 per stride).

    This leg's heel strike is where gy crosses zero after its swing peak (the
    end of the forward swing); the other leg's comes half a stride later. These
    are the warm-start pseudo-labels and the baseline's step times.
    """
    gy = np.asarray(gy, dtype=float)
    peaks, period = swing_peaks(gy)
    steps = []
    for i, p in enumerate(peaks):
        local = (peaks[i + 1] - p) if i + 1 < len(peaks) else (p - peaks[i - 1]) if i else (period or FS)
        after = gy[p:p + int(0.6 * local)]
        down = np.flatnonzero((after[:-1] > 0) & (after[1:] <= 0))
        hs = p + (down[0] + 1 if len(down) else int(0.23 * local))
        steps += [hs, hs + local // 2]
    return np.array([s for s in steps if s < len(gy)], dtype=int)
