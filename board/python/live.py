"""Live classification: rolling 2 s buffer -> despike -> features -> both models.

Pure Python + numpy, no Arduino imports, so the laptop can run exactly the
code the board runs (laptop/live_test.py). main.py feeds it samples from the
Bridge; the slow part (classify) runs on a worker thread, not in the Bridge
handler.

    live = LiveClassifier()            # loads models/model1.json + models/model2_rules.py
    snapshot = live.add(sample)        # every sample (cheap); a snapshot every 0.5 s
    if snapshot is not None:
        result = live.classify(snapshot)   # -> (prediction dict, unsmoothed model2) or None

The prediction dict is the ME193/ceci/steps/pred format (see CLAUDE.md):
    {"t": <ms, newest sample in the window>, "model1": ..., "model2": ..., "cadence_hz": ...}
"""

import collections
import importlib.util
import json
from pathlib import Path

import numpy as np

import features as F

MODELS_DIR = Path(__file__).resolve().parent / "models"
CLASSIFY_EVERY = F.FS // 2      # samples between predictions (0.5 s)
VOTE = 3                        # Model 2 majority vote over this many predictions
FIELDS = ("t",) + F.AXES


def load_models(models_dir=MODELS_DIR):
    model1 = json.loads((Path(models_dir) / "model1.json").read_text())
    spec = importlib.util.spec_from_file_location("model2_rules", Path(models_dir) / "model2_rules.py")
    model2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model2)
    return model1, model2


def majority(labels):
    """Most common label; on a tie the most recent of the tied labels wins."""
    counts = collections.Counter(labels)
    top = max(counts.values())
    return next(label for label in reversed(labels) if counts[label] == top)


class LiveClassifier:
    def __init__(self, models_dir=MODELS_DIR):
        self.model1, self.model2 = load_models(models_dir)
        # The 2 s window plus one sample on each side, so despike() sees both
        # neighbours of every sample in the window (costs 20 ms of latency).
        self.buf = collections.deque(maxlen=F.WINDOW + 2)
        self.since = 0
        self.votes = collections.deque(maxlen=VOTE)

    def reset(self):
        self.buf.clear()
        self.votes.clear()
        self.since = 0

    def add(self, sample):
        """Add one sample (dict with t, ax..gz). Returns a buffer snapshot every
        CLASSIFY_EVERY samples once the buffer is full, else None."""
        if self.buf and sample["t"] <= self.buf[-1][0]:
            self.reset()                      # t went backwards: the sketch restarted
        self.buf.append([sample[k] for k in FIELDS])
        self.since += 1
        if self.since < CLASSIFY_EVERY or len(self.buf) < self.buf.maxlen:
            return None
        self.since = 0
        return np.array(self.buf, dtype=float)

    def classify(self, snapshot):
        """(prediction dict, unsmoothed model2 label), or None if the window has a gap."""
        t, x = snapshot[:, 0], snapshot[:, 1:]
        if t[-2] - t[1] != (F.WINDOW - 1) * F.SAMPLE_MS:
            return None
        clean, _ = F.despike(x)
        feats = F.window_features(clean[1:-1])
        model1 = F.model1_predict(feats, self.model1)
        raw2 = self.model2.predict(feats)
        self.votes.append(raw2)
        model2 = majority(list(self.votes))
        cadence = 0.0 if model2 == "standing" else round(feats["cadence_hz"], 2)
        return {"t": int(t[-2]), "model1": model1, "model2": model2, "cadence_hz": cadence}, raw2
