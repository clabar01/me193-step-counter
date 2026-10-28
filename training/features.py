"""Phase 4: cut every recording in data/raw/ into 2 s windows (1 s step) and
compute all Model 1 and Model 2 features. Writes data/processed/windows.csv.

The feature code itself lives in board/python/features.py (shared with the
board), so training and live classification compute exactly the same thing.
Per recording: despike() over the whole recording first, then window.

Only data/raw/ is read; simulator data in data/sim/ is never used.
Air-biking becomes the "biking" class; its original label stays in the
"activity" column so it can still be checked separately.

Usage:
    python training/features.py
"""

from pathlib import Path

import pandas as pd

from shared import REPO, shared_features

F = shared_features()
RAW_DIR = REPO / "data" / "raw"
OUT = REPO / "data" / "processed" / "windows.csv"


def windows_for_recording(path):
    df = pd.read_csv(path).sort_values("t").reset_index(drop=True)
    clean, fixed = F.despike(df[list(F.AXES)].to_numpy())
    t = df["t"].to_numpy()
    rows = []
    for start in F.window_starts(t):
        feats = F.window_features(clean[start:start + F.WINDOW])
        activity = df["label"].iloc[start]
        rows.append({
            "subject": df["subject"].iloc[start],
            "recording": path.name,
            "activity": activity,
            "label": F.merge_label(activity),
            "t_start": int(t[start]),
            "t_end": int(t[start + F.WINDOW - 1]),
            **feats,
        })
    return rows, len(fixed), len(df)


def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"no recordings in {RAW_DIR.relative_to(REPO)}/")
    rows = []
    for path in files:
        recording_rows, n_fixed, n_samples = windows_for_recording(path)
        rows += recording_rows
        print(f"{path.name}: {n_samples} samples, {n_fixed} despiked, {len(recording_rows)} windows")

    windows = pd.DataFrame(rows)
    windows.to_csv(OUT, index=False)
    print(f"\nsaved {OUT.relative_to(REPO)}: {len(windows)} windows, "
          f"{len(windows.columns) - 6} features")
    print(pd.crosstab(windows["label"], windows["subject"], margins=True).to_string())
    missing = [c for c in F.CLASSES if c not in set(windows["label"])]
    if missing:
        print(f"\nclasses with no data yet: {', '.join(missing)}")


if __name__ == "__main__":
    main()
