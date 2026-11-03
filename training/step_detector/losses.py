"""Losses for the two training stages. All take padded batches:
y (batch, N) model outputs, mask (batch, N) 1 for real samples, 0 for padding.

The idea: the output is a "step mass". Each step should put a total of about 1
inside one short bump (STEP_SAMPLES = 100 ms), and nothing anywhere else. Then
the plain sum of the output is the step count, and so is the number of bumps.
See docs/step_detector.md for why each piece is there.
"""

import torch
import torch.nn.functional as F

FS = 50
STEP_SAMPLES = 5            # 100 ms: one step's mass should fit inside this
PEAK_WINDOW = 13            # 0.25 s: at most one step fits in a window this long
POS_WEIGHT = 20.0           # warm start: a step sample counts this much more than a non-step one


def moving_sum(y, k):
    """Sum of y over a centred window of k samples, same length as y."""
    return F.avg_pool1d(y.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=True).squeeze(1) * k


def count_loss(y, mask, counted):
    """Per trial (sum of outputs - counted steps)^2, averaged over the batch.

    The only label is how many steps there were, so the model is free to put
    them wherever the sensors say; it only has to add up to the right number.
    """
    total = (y * mask).sum(dim=1)
    return ((total - counted) ** 2).mean()


def peak_penalty(y, mask):
    """Keep each step one sharp bump. Two parts, both zero for a clean output:

    - spread: in every 0.25 s window, the mass that is NOT within the 100 ms
      around the window's peak. A low, flat "step rate" while walking (which
      adds up to the right count with no countable steps at all) is mostly
      spread, so it costs a lot; a step packed into 100 ms costs nothing.
    - crowding: (mass in a 0.25 s window - 1)^2 wherever it exceeds 1. Two
      steps can't happen within 0.25 s, so this stops the model from piling
      two steps' worth on one leg's step and dropping the other leg's.

    Summed over each trial, averaged over the batch.
    """
    y = y * mask
    window = moving_sum(y, PEAK_WINDOW)
    best_bump = F.max_pool1d(moving_sum(y, STEP_SAMPLES).unsqueeze(1), PEAK_WINDOW, stride=1,
                             padding=PEAK_WINDOW // 2).squeeze(1)
    spread = (F.relu(window - best_bump) * mask).sum(dim=1)
    crowding = (F.relu(window - 1) ** 2 * mask).sum(dim=1)
    return (spread + crowding).mean()


def pseudo_targets(n, steps, device):
    """Warm-start target for one trial: 1/5 on each of the 5 samples (100 ms)
    around each pseudo-step, so every step adds up to 1, and 0 elsewhere. The
    step samples are up-weighted because they are rare (~1 in 5 while walking).

    Not 1 at the exact step sample: the gyro detector's step time is only good
    to a sample or two, and a target of 1 trains outputs that saturate near 1
    over several samples (3-4x too much mass per step), which the count loss
    then has to undo (see docs/step_detector.md)."""
    target = torch.zeros(n, device=device)
    weight = torch.ones(n, device=device)
    half = STEP_SAMPLES // 2
    for s in steps:
        lo, hi = max(0, s - half), min(n, s + half + 1)
        target[lo:hi] = 1.0 / STEP_SAMPLES
        weight[lo:hi] = POS_WEIGHT
    return target, weight


def pseudo_label_loss(y, mask, target, weight):
    """Weighted binary cross-entropy against the gyro detector's step times."""
    bce = F.binary_cross_entropy(y.clamp(1e-6, 1 - 1e-6), target, reduction="none")
    w = weight * mask
    return (bce * w).sum() / w.sum()
