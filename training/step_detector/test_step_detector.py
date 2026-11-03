"""Checks for the step-detector pipeline (no data files needed).

Run: python training/step_detector/test_step_detector.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counting import event_f1, peak_steps  # noqa: E402
from data import CHANNELS, US_CLIP_MM, align_ultrasonic, clip_ultrasonic, despike, preprocess  # noqa: E402
from losses import count_loss, peak_penalty, pseudo_targets  # noqa: E402
from model import StepCNN, n_parameters, receptive_field  # noqa: E402
from strides import step_times, swing_peaks  # noqa: E402

BASE = np.tile([1.0, 0.0, 0.0, 0.0, 50.0, 0.0], (5, 1))


def test_despike_replaces_isolated_glitch():
    # The real corrupted read from the gait project (ceci_air-biking_20261023-181650.csv)
    x = BASE.copy()
    x[2] = [-12.2, -0.13, 6.71, 401.96, 0.48, 754.34]
    clean, bad = despike(x)
    assert bad.tolist() == [2]
    assert np.allclose(clean[2], BASE[2])


def test_despike_keeps_single_axis_impact_peak():
    x = BASE.copy()
    x[2, 0] = 5.5               # like a 5.5 g heel strike: one accel axis, gyro smooth
    assert despike(x)[1].tolist() == []


def test_ultrasonic_moves_back_one_row():
    assert align_ultrasonic([10, 20, 30, 40], 1).tolist() == [20, 30, 40, 40]
    assert align_ultrasonic([10, 20], 0).tolist() == [10, 20]


def test_far_echoes_and_timeouts_read_as_no_echo():
    mm, hit = clip_ultrasonic(np.array([30, 149, 150, 560, 1000]), np.array([1, 1, 1, 1, 0]))
    assert mm.tolist() == [30, 149, US_CLIP_MM, US_CLIP_MM, US_CLIP_MM]
    assert hit.tolist() == [1, 1, 0, 0, 0]


def test_preprocess_shape_and_order():
    df = pd.DataFrame({"t_ms": [0, 20, 40], "ax": 1.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 5.0, "gz": 0.0,
                       "us_mm": [1000, 25, 700], "us_hit": [0, 1, 1]})
    x, n_bad = preprocess(df, lag=1)
    assert x.shape == (3, len(CHANNELS)) and n_bad == 0
    assert x[:, CHANNELS.index("us_mm")].tolist() == [25, US_CLIP_MM, US_CLIP_MM]


def test_gyro_detector_counts_two_steps_per_swing():
    t = np.arange(0, 12.5, 0.02)                        # ends just after the 20th step
    gy = 130 * np.sin(2 * np.pi * t / 1.2) * (t < 12)   # 10 strides of 1.2 s
    assert len(swing_peaks(gy)[0]) == 10
    assert len(step_times(gy)) == 20
    assert len(step_times(np.random.default_rng(0).normal(0, 2, 600))) == 0     # standing still


def test_peak_count_merges_a_spread_step_and_keeps_gaps():
    y = np.zeros(200)
    y[50:55] = 0.2              # one step spread over 100 ms: mass 1
    y[100], y[103] = 0.6, 0.6   # 60 ms apart: one step, not two
    y[150] = 0.3                # too small to count
    assert peak_steps(y).tolist() in ([52, 101], [52, 102])


def test_event_f1():
    assert event_f1([10, 50, 90], [11, 52, 200])[2] == 2 / 3
    assert event_f1([], [])[2] == 1.0


def test_losses_reward_one_bump_per_step():
    mask = torch.ones(1, 100)
    clean = torch.zeros(1, 100)
    clean[0, 18:23] = 0.2                       # one step, inside 100 ms
    flat = torch.full((1, 100), 0.01)           # the same mass spread as a "rate"
    doubled = torch.zeros(1, 100)
    doubled[0, 18:23] = 0.4                     # two steps' mass on one step
    assert abs(count_loss(clean, mask, torch.tensor([1.0])).item()) < 1e-6
    assert peak_penalty(clean, mask) < 1e-6
    assert peak_penalty(flat, mask) > 0.5
    assert peak_penalty(doubled, mask) > 0.5


def test_pseudo_targets_add_up_to_one_per_step():
    target, weight = pseudo_targets(100, [20, 60], torch.device("cpu"))
    assert abs(target.sum().item() - 2.0) < 1e-6
    assert weight.max().item() > 1


def test_model_size_and_length():
    model = StepCNN()
    assert n_parameters(model) < 20_000
    assert receptive_field() == 121
    with torch.no_grad():
        out = model(torch.zeros(2, len(CHANNELS), 333))
    assert out.shape == (2, 333) and float(out.min()) >= 0 and float(out.max()) <= 1


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
