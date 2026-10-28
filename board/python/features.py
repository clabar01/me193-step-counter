"""Shared feature code for training (laptop) and live classification (board).

This module lives in board/python/ so it is deployed with the board app;
laptop scripts import it from there. Both sides must compute features with
exactly this code so the models see the same inputs in training and live.

Samples are an (N, 6) array with columns AXES, accel in g and gyro in deg/s,
body frame, 50 Hz.

Pipeline order: despike() FIRST, then any feature (including Model 1's peak
accel). Run despike() on the continuous stream (a whole recording, or a rolling
buffer on the board), not on each 2 s window by itself, so samples at window
edges still have both neighbours. Then:

    clean, _ = despike(samples)
    for start in window_starts(t):
        feats = window_features(clean[start:start + WINDOW])
        label1 = model1_predict(feats, model1)          # models/model1.json
        label2 = model2_rules.predict(feats)            # models/model2_rules.py

live.py does exactly this on the board. Needs numpy (in python/requirements.txt).
"""

import numpy as np

AXES = ("ax", "ay", "az", "gx", "gy", "gz")
FS = 50
WINDOW = 2 * FS                      # 2 s windows...
STEP = 1 * FS                        # ...every 1 s
SAMPLE_MS = 1000 // FS

# Labels the models use. Air-biking is recorded with its own label but is the
# same class as biking.
CLASSES = ("standing", "walking", "running", "biking")
LABEL_MERGE = {"air-biking": "biking"}

# Cadence: strongest frequency of the signed gy (thigh swing, stride rate) in
# this band. 2 s of data alone resolves only 0.5 Hz, so the window is
# zero-padded to CADENCE_NFFT to interpolate the peak.
CADENCE_BAND_HZ = (0.3, 4.0)
CADENCE_NFFT = 1024
CADENCE_MIN_STD_DPS = 5.0            # below this gy is "not swinging": cadence 0
CADENCE_PEAK_HZ = 0.25

MODEL1_FEATURES = ("peak_ax", "peak_ay", "peak_az")

# A sample is a spike if, on any axis, it is more than SPIKE from BOTH
# neighbours while the neighbours are within AGREE of each other. Tuned on the
# first real recordings (2026-10-23): the one corrupted I2C read jumped 11 g and
# 750 deg/s for a single sample on four axes. At 3 g / 300 deg/s the filter
# also removed three single-axis accel peaks (4.0-5.5 g in walking, 3.5 g in
# air-biking) whose gyro stayed smooth, which look like real heel-strike/kick
# impacts. Model 1 is peak accel, so those must survive: 6 g / 500 deg/s keeps
# them and still catches the glitch. Re-check this when running data exists.
SPIKE = np.array([6.0, 6.0, 6.0, 500.0, 500.0, 500.0])
AGREE = SPIKE / 2


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


def merge_label(label):
    """Recording label -> model class (air-biking -> biking)."""
    return LABEL_MERGE.get(label, label)


def window_starts(t):
    """Start indices of full 2 s windows, 1 s apart, that contain no gap in t (ms)."""
    t = np.asarray(t)
    span = (WINDOW - 1) * SAMPLE_MS
    return [s for s in range(0, len(t) - WINDOW + 1, STEP) if t[s + WINDOW - 1] - t[s] == span]


def cadence(gy):
    """(stride frequency Hz, peak strength 0-1) from one window of signed gy."""
    gy = np.asarray(gy, dtype=float)
    if gy.std() < CADENCE_MIN_STD_DPS:
        return 0.0, 0.0
    x = (gy - gy.mean()) * np.hanning(len(gy))
    power = np.abs(np.fft.rfft(x, CADENCE_NFFT)) ** 2
    freqs = np.fft.rfftfreq(CADENCE_NFFT, 1 / FS)
    band = (freqs >= CADENCE_BAND_HZ[0]) & (freqs <= CADENCE_BAND_HZ[1])
    i = np.argmax(np.where(band, power, -1))
    total = power[band].sum()
    # strength: share of the band's power within +/-CADENCE_PEAK_HZ of the peak
    # (zero-padding spreads one peak over many bins)
    near = band & (np.abs(freqs - freqs[i]) <= CADENCE_PEAK_HZ)
    return float(freqs[i]), float(power[near].sum() / total) if total > 0 else 0.0


def window_features(w):
    """All Model 1 and Model 2 features for one despiked (WINDOW, 6) window."""
    w = np.asarray(w, dtype=float)
    acc, gyr = w[:, :3], w[:, 3:]
    f = {}

    # Model 1: peak accel of each axis (largest |value| in the window)
    for i, axis in enumerate(("ax", "ay", "az")):
        f[f"peak_{axis}"] = float(np.abs(acc[:, i]).max())

    # Model 2 adds everything below
    for i, axis in enumerate(AXES):
        f[f"mean_{axis}"] = float(w[:, i].mean())
        f[f"std_{axis}"] = float(w[:, i].std())

    amag = np.linalg.norm(acc, axis=1)
    gmag = np.linalg.norm(gyr, axis=1)
    f["amag_mean"] = float(amag.mean())
    f["amag_std"] = float(amag.std())
    f["amag_max"] = float(amag.max())
    f["amag_min"] = float(amag.min())
    f["gmag_mean"] = float(gmag.mean())
    f["gmag_max"] = float(gmag.max())
    f["gyro_energy"] = float((gyr ** 2).sum(axis=1).mean())

    # Gravity direction: unit vector of the mean accel (standing ~ +x,
    # sitting ~ +z, lying on your back with legs up ~ -x)
    g = acc.mean(axis=0)
    g = g / (np.linalg.norm(g) or 1.0)
    f["grav_x"], f["grav_y"], f["grav_z"] = (float(v) for v in g)

    f["cadence_hz"], f["cadence_strength"] = cadence(w[:, AXES.index("gy")])
    return f


def model1_predict(features, model):
    """Model 1: thresholds on one peak-accel feature (models/model1.json).

    model["classes"] are ordered by increasing feature value and
    model["thresholds"] has one fewer entry: class i covers values up to
    thresholds[i].
    """
    v = features[model["feature"]]
    for threshold, cls in zip(model["thresholds"], model["classes"]):
        if v <= threshold:
            return cls
    return model["classes"][-1]
