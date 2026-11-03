"""Checks for the step-length code (no data files needed).

Run: python training/step_length/test_step_length.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, IX, step_features, walking_bout  # noqa: E402
from train import StepLengthMLP, baseline_terms, fit_mlp, fit_scale, trial_loss  # noqa: E402


def fake_walk(strides=10, period_s=1.2, stand_s=4.0, extra_twitch_s=None):
    """(N, 8) inputs: standing, a clean gy swing per stride, standing (optional late leg twitch)."""
    t = np.arange(0, stand_s * 2 + strides * period_s, 0.02)
    walking = (t >= stand_s) & (t < stand_s + strides * period_s)
    x = np.zeros((len(t), 8), dtype=np.float32)
    x[:, IX["ax"]] = 0.95
    x[:, IX["gy"]] = 130 * np.sin(2 * np.pi * (t - stand_s) / period_s) * walking
    if extra_twitch_s is not None:
        twitch = np.abs(t - extra_twitch_s) < 0.15
        x[twitch, IX["gy"]] = 90 * np.cos(np.pi * (t[twitch] - extra_twitch_s) / 0.3)
    x[:, IX["us_mm"]] = 150
    return x


def test_walking_bout_drops_a_late_twitch():
    peaks = np.array([100, 160, 220, 280, 340, 700])
    assert walking_bout(peaks).tolist() == [100, 160, 220, 280, 340]


def test_step_features_two_per_stride_and_swing_angle():
    f, steps = step_features(fake_walk(strides=10, extra_twitch_s=24.0))
    assert f.shape[1] == len(FEATURES)
    assert 19 <= len(steps) <= 21                      # the twitch 4 s after stopping is not a step
    swing = np.median(f[:, FEATURES.index("swing_deg")])
    expected = 2 * 130 * 1.2 / (2 * np.pi)              # range of the integral of a 130 deg/s sine
    assert abs(swing - expected) < 3


def test_fit_scale_is_exact_when_possible():
    terms = [np.ones(20), np.ones(30)]
    assert abs(fit_scale(terms, [15.0, 22.5]) - 0.75) < 1e-9


def test_baseline_terms():
    f = np.zeros((3, len(FEATURES)))
    f[:, FEATURES.index("acc_range_g")] = 16.0
    f[:, FEATURES.index("swing_deg")] = 60.0
    assert np.allclose(baseline_terms("weinberg", f, 1.0), 2.0)
    assert np.allclose(baseline_terms("pendulum", f, 0.9), 0.9)       # 2 L sin(30 deg)


def test_trial_loss_is_relative():
    assert abs(trial_loss(torch.tensor([11.0, 33.0]), torch.tensor([10.0, 30.0])).item() - 0.01) < 1e-6


def test_mlp_is_small_and_positive():
    model = StepLengthMLP()
    assert sum(p.numel() for p in model.parameters()) == 385
    with torch.no_grad():
        assert (model(torch.randn(50, len(FEATURES)) * 5) > 0).all()


def test_mlp_learns_a_constant_step_from_trial_sums():
    rng = np.random.default_rng(0)
    feats = [rng.normal(size=(n, len(FEATURES))) for n in (20, 30, 40)]
    model, mean, std = fit_mlp(feats, [16.0, 24.0, 32.0], seed=0)          # 0.8 m every step
    with torch.no_grad():
        total = model(torch.tensor((feats[1] - mean) / std, dtype=torch.float32)).sum().item()
    assert abs(total - 24.0) < 0.5


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
