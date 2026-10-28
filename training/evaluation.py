"""Train/test splits shared by model1.py and model2.py, so both models are
always evaluated on exactly the same held-out windows.

- 2+ subjects: leave-one-subject-out (one fold per subject).
- 1 subject:   time-based split inside each recording: first TRAIN_FRACTION of
  its windows train, the rest test. The window straddling the boundary is
  dropped because neighbouring windows share half their samples. This is
  optimistic (same person, same session) and only a stand-in until there are
  more subjects.
"""

import numpy as np

TRAIN_FRACTION = 0.7


def splits(windows):
    """Return (method description, [(fold name, train index, test index), ...])."""
    subjects = sorted(windows["subject"].unique())
    if len(subjects) >= 2:
        folds = [(f"test on {s}", windows.index[windows["subject"] != s].to_numpy(),
                  windows.index[windows["subject"] == s].to_numpy()) for s in subjects]
        return f"leave-one-subject-out ({len(subjects)} subjects)", folds

    train, test = [], []
    for _, rec in windows.sort_values("t_start").groupby("recording"):
        idx = rec.index.to_numpy()
        cut = int(round(TRAIN_FRACTION * len(idx)))
        train += list(idx[:max(cut - 1, 0)])    # drop the window that overlaps the test part
        test += list(idx[cut:])
    method = (f"time split within each recording (first {TRAIN_FRACTION:.0%} train, rest test; "
              f"only 1 subject: {subjects[0]})")
    return method, [("time split", np.array(train), np.array(test))]


def classes_present(windows, order):
    present = set(windows["label"])
    return [c for c in order if c in present]
