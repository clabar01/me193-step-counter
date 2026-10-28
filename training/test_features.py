"""Checks for the shared feature code (board/python/features.py).

Run: python training/test_features.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "board" / "python"))
from features import despike  # noqa: E402

BASE = np.tile([1.0, 0.0, 0.0, 0.0, 50.0, 0.0], (5, 1))


def test_replaces_isolated_glitch():
    # The real corrupted read from ceci_air-biking_20261023-181650.csv, t=1630800
    x = BASE.copy()
    x[2] = [-12.2, -0.13, 6.71, 401.96, 0.48, 754.34]
    clean, bad = despike(x)
    assert bad.tolist() == [2]
    assert np.allclose(clean[2], BASE[2])


def test_keeps_single_axis_impact_peak():
    # Like the 5.5 g walking heel strike: one accel axis, gyro smooth
    x = BASE.copy()
    x[2, 0] = 5.5
    assert despike(x)[1].tolist() == []


def test_keeps_impact_spread_over_samples():
    x = BASE.copy()
    x[1:4, 0] = [4.0, 9.0, 4.5]
    assert despike(x)[1].tolist() == []


def test_keeps_spike_when_neighbours_disagree():
    x = BASE.copy()
    x[1, 0], x[2, 0], x[3, 0] = 1.0, 9.0, 5.0
    assert despike(x)[1].tolist() == []


def test_leaves_first_and_last_sample():
    x = np.vstack([[20, 0, 0, 0, 0, 0], BASE, [20, 0, 0, 0, 0, 0]])
    assert despike(x)[1].tolist() == []


def test_short_input():
    assert despike(BASE[:2])[1].tolist() == []


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
