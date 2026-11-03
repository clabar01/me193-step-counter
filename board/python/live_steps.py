"""Live step counting and step length on the board, with numpy only.

The two trained models (exported by scripts/export_models.py to
models/step_detector.npz and models/step_length.npz) run here exactly as in
training (training/step_detector/, training/step_length/):

  every sample      add() appends it to a rolling buffer (BUFFER samples, 6 s)
  every 0.25 s      process():
    1. preprocess the buffer like training: despike the IMU, move each ping
       back one row (us_lag_samples), clip the ultrasonic at 150 mm, z-score
    2. run the CNN over the whole buffer -> step output, and its 100 ms sum
       ("step mass")
    3. FINALIZE the samples whose output can't change any more: the CNN looks
       1.2 s (60 samples) each way, so a sample is final once 60 newer samples
       (plus a few for despiking, alignment and the 100 ms sum) are in. Each
       sample is finalized exactly once, so overlapping windows never count a
       step twice.
    4. a finalized sample is a step if its mass is >= 0.5, it is the largest
       within +-0.25 s, and the last step was >= 0.25 s earlier (the same rule
       as training/step_detector/counting.py)
    5. a step's length is computed when the NEXT step arrives (its features
       need the step time), or after STEP_TIMEOUT_S if none does (the last step
       of a walk uses the previous step time, as in training)

So a step is counted ~1.3-1.5 s after it happens: the CNN needs 1.2 s of
"future" around each sample, plus up to 0.25 s until the next processing run.
test_live_steps.py (training/step_detector/) checks this gives the same steps
as the offline pipeline on the real trials.
"""

import collections
from pathlib import Path

import numpy as np

FS = 50
BUFFER = 300                 # 6 s
RF_HALF = 60                 # CNN receptive field: 121 samples = +-60
MASS_K = 5                   # step mass = 100 ms moving sum
PEAK_HALF = 6                # a step is the largest mass within +-6 samples (0.25 s window)
PEAK_HEIGHT = 0.5
MIN_GAP = 13                 # samples between steps (0.25 s)
FINAL_LAG = RF_HALF + 2 + MASS_K // 2 + 1   # newest sample index - this = newest final mass
STEP_TIMEOUT_S = 1.5         # no next step this long -> finish the last step's length
CADENCE_STEPS = 6            # cadence from the median of the last this-many step intervals
CADENCE_IDLE_S = 2.0         # no step this long -> cadence 0
AXES = 6

# Same glitch filter as training/step_detector/data.py
SPIKE = np.array([6.0, 6.0, 6.0, 500.0, 500.0, 500.0])
AGREE = SPIKE / 2


def despike(x):
    x = np.array(x, dtype=float)
    if len(x) < 3:
        return x
    prev, cur, nxt = x[:-2], x[1:-1], x[2:]
    far = (np.abs(cur - prev) > SPIKE) & (np.abs(cur - nxt) > SPIKE)
    agree = np.abs(prev - nxt) <= AGREE
    bad = np.flatnonzero((far & agree).any(axis=1)) + 1
    x[bad] = (x[bad - 1] + x[bad + 1]) / 2
    return x


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


class StepCNN:
    """numpy forward pass of training/step_detector/model.py (dilated Conv1d, same padding)."""

    def __init__(self, path):
        d = np.load(path)
        self.mean, self.std = d["mean"].astype(float), d["std"].astype(float)
        self.clip_mm = float(d["us_clip_mm"])
        self.layers = [(d[f"conv{i}_w"], d[f"conv{i}_b"], int(dil)) for i, dil in enumerate(d["dilations"])]
        self.head_w, self.head_b = d["head_w"], d["head_b"]

    def __call__(self, x):
        """x: (8, T) z-scored -> (T,) step output in 0..1."""
        h = x.astype(np.float32)
        for w, b, dil in self.layers:
            out_ch, in_ch, k = w.shape
            pad = dil * (k - 1) // 2
            hp = np.pad(h, ((0, 0), (pad, pad)))
            t = h.shape[1]
            cols = np.stack([hp[:, j * dil:j * dil + t] for j in range(k)], axis=1)     # (in, k, T)
            h = np.maximum(w.reshape(out_ch, -1) @ cols.reshape(in_ch * k, t) + b[:, None], 0)
        z = self.head_w @ h + self.head_b[:, None]
        return 1 / (1 + np.exp(-z[0]))


class StepLengthMLP:
    """numpy forward pass of training/step_length/train.py's StepLengthMLP."""

    def __init__(self, path):
        d = np.load(path)
        self.features = list(d["features"])
        self.mean, self.std = d["mean"].astype(float), d["std"].astype(float)
        n = len([k for k in d.files if k.startswith("lin") and k.endswith("_w")])
        self.layers = [(d[f"lin{i}_w"], d[f"lin{i}_b"]) for i in range(n)]

    def __call__(self, f):
        h = (np.asarray(f, dtype=float) - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = w @ h + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return float(np.log1p(np.exp(h[0])))         # softplus


class LiveSteps:
    def __init__(self, models_dir):
        models_dir = Path(models_dir)
        self.cnn = StepCNN(models_dir / "step_detector.npz")
        self.mlp = StepLengthMLP(models_dir / "step_length.npz")
        self.buf = collections.deque(maxlen=BUFFER)    # (index, t, ax..gz, us_mm, us_hit)
        self.n = 0                                     # samples added so far
        self.mass = {}                                 # finalized index -> step mass (recent only)
        self.final_upto = -1
        self.decided_upto = -1
        self.step_idx = []                             # recent step indices (for cadence / stride)
        self.pending = None                            # (index, t) of the step waiting for its length
        self.reset()

    def reset(self):
        """Zero the totals (the detector keeps running)."""
        self.steps = 0
        self.distance_m = 0.0
        self.last_length = None

    def add(self, s):
        self.buf.append((self.n, s["t"], s["ax"], s["ay"], s["az"], s["gx"], s["gy"], s["gz"],
                         s["us_mm"], s["us_hit"]))
        self.n += 1

    def _inputs(self, arr):
        imu = despike(arr[:, 2:8])
        us_mm = np.concatenate([arr[1:, 8], arr[-1:, 8]])      # each ping back one row
        us_hit = np.concatenate([arr[1:, 9], arr[-1:, 9]])
        near = (us_hit == 1) & (us_mm < self.cnn.clip_mm)
        us_mm = np.where(near, us_mm, self.cnn.clip_mm)
        return np.column_stack([imu, us_mm, near.astype(float)])

    def process(self):
        """Run on the buffer. Returns what changed since the last call:
        {"p": [mass of each newly final sample], "p_t0": t of the first,
         "steps": [{"t", "n"}], "lengths": [{"t", "length_m"}], "t": newest sample t}."""
        out = {"p": [], "p_t0": None, "steps": [], "lengths": [], "t": self.buf[-1][1] if self.buf else None}
        if len(self.buf) < 2 * FINAL_LAG + 1:
            return out
        arr = np.array(self.buf, dtype=float)
        first, ts = int(arr[0, 0]), arr[:, 1]
        x = self._inputs(arr)
        y = self.cnn(((x - self.cnn.mean) / self.cnn.std).T)
        mass = np.convolve(y, np.ones(MASS_K), "same")

        lo = max(self.final_upto + 1, first + FINAL_LAG)
        hi = first + len(arr) - 1 - FINAL_LAG
        for i in range(lo, hi + 1):
            self.mass[i] = float(mass[i - first])
        if hi >= lo:
            out["p"] = [round(self.mass[i], 3) for i in range(lo, hi + 1)]
            out["p_t0"] = int(ts[lo - first])
            self.final_upto = hi

        # decide each sample once its +-PEAK_HALF neighbours are final
        for j in range(self.decided_upto + 1, self.final_upto - PEAK_HALF + 1):
            m = self.mass.get(j, 0.0)
            if m >= PEAK_HEIGHT and (not self.step_idx or j - self.step_idx[-1] >= MIN_GAP):
                left = max((self.mass.get(i, 0.0) for i in range(j - PEAK_HALF, j)), default=0.0)
                right = max(self.mass.get(i, 0.0) for i in range(j + 1, j + PEAK_HALF + 1))
                if m > left and m >= right:
                    self._on_step(j, int(ts[j - first]), arr, x, first, out)
            self.decided_upto = j
        for i in [i for i in self.mass if i < self.decided_upto - 2 * PEAK_HALF]:
            del self.mass[i]

        # the last step of a walk: no next step within STEP_TIMEOUT_S
        if self.pending and self.final_upto - self.pending[0] > STEP_TIMEOUT_S * FS:
            s = self.pending[0]
            prev = [p for p in self.step_idx if p < s]
            nxt = s + (s - prev[-1] if prev and s - prev[-1] < STEP_TIMEOUT_S * FS else self._stride() // 2)
            self._finish(s, nxt, arr, x, first, out)
        return out

    def _stride(self):
        recent = np.diff(self.step_idx[-(CADENCE_STEPS + 1):])
        recent = recent[recent < STEP_TIMEOUT_S * FS]
        return int(2 * np.median(recent)) if len(recent) else 60

    def _on_step(self, j, t, arr, x, first, out):
        self.step_idx = (self.step_idx + [j])[-20:]
        self.steps += 1
        out["steps"].append({"t": t, "n": self.steps})
        if self.pending:
            self._finish(self.pending[0], j, arr, x, first, out)
        self.pending = (j, t)

    def _finish(self, s, nxt, arr, x, first, out):
        """Step length for the step at index s, given the next step's index."""
        f = self.step_features(x, s - first, nxt - first, self._stride())
        length = self.mlp(f)
        self.distance_m += length
        self.last_length = length
        out["lengths"].append({"t": int(arr[s - first, 1]), "length_m": round(length, 3)})
        self.pending = None

    @staticmethod
    def step_features(x, s, nxt, stride):
        """Same features as training/step_length/features.py, for buffer rows s..nxt."""
        nxt = min(max(nxt, s + 1), len(x) - 1)
        acc = np.linalg.norm(x[:, :3], axis=1)
        gy = x[:, 4]
        a, b = max(0, s - stride // 2), min(len(x), s + stride // 2)
        angle = np.cumsum(gy[a:b]) / FS
        angle -= np.linspace(angle[0], angle[-1], len(angle))
        within = slice(s, nxt)
        near = x[within, 7] == 1
        us = x[within, 6]
        return [(nxt - s) / FS,
                angle.max() - angle.min(),
                acc[within].max() - acc[within].min(),
                us[near].min() if near.any() else 150.0,
                longest_run(near) * 1000 / FS]

    def cadence_spm(self):
        """Steps per minute from recent steps; 0 once no step for CADENCE_IDLE_S."""
        if len(self.step_idx) < 2 or self.final_upto - self.step_idx[-1] > CADENCE_IDLE_S * FS:
            return 0.0
        gaps = np.diff(self.step_idx[-(CADENCE_STEPS + 1):])
        gaps = gaps[gaps < STEP_TIMEOUT_S * FS]
        return round(60 * FS / float(np.median(gaps)), 1) if len(gaps) else 0.0
