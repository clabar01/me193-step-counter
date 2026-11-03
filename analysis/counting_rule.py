"""Choose the CNN's counting rule using the training trials only.

The CNN's output sum is close to the hand count, but counting its peaks above
0.5 misses steps, nearly always the last step of a walk (the output there is ~0,
not a weak peak). Candidate rules:

  (a) peaks      peaks of the 100 ms step mass above a lower threshold h
  (b) sum        round(sum of the output): what the count loss trains
  (c) edge       peaks above 0.5, plus one step at the end of each walking bout
                 of at least k steps (a bout ends when no step follows for 1.5 s,
                 the board's STEP_TIMEOUT_S)
      edge-gyro  as (c), but the edge step is added only if the gyro stride
                 detector (strides.py) has a step 0.5-1.5 step intervals
                 before the bout's first / after its last peak

Scoring every rule on the committed model's own training trials would be unfair:
the count loss fitted exactly sum = counted on those trials, so (b) scores
perfectly there by construction. So the CNN is retrained K times inside
the training set, each time without one fold of training trials, with the
committed run's settings. Every rule is then scored on the trials each model
never saw. Each fold model picks its kept epoch on its own training trials, so
neither the validation trials nor Session C play any part. The winner is
frozen in training/step_detector/counting.py (count_steps) and then reported
on the validation trials by analysis/evaluate.py.

Usage: python analysis/counting_rule.py         (~3 min; writes analysis/counting_rule.md)
"""

import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training" / "step_detector"))
import train as det  # noqa: E402
from counting import EDGE_GAP_S, FS, PEAK_GAP_S, bouts, count_steps, step_mass  # noqa: E402
from data import CHANNELS, load_trials  # noqa: E402
from model import StepCNN  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402
from strides import step_times  # noqa: E402

OUT = REPO / "analysis" / "counting_rule.md"
FOLDS = 7                     # 14 training trials -> 2 held out per fold
SEEDS = (0, 1)                # two different fold assignments and training seeds
THRESHOLDS = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
MIN_BOUTS = (1, 2, 3, 4, 6)


def peaks(y, h):
    return find_peaks(step_mass(y), height=h, distance=int(PEAK_GAP_S * FS))[0]


def edge_count(y, k):
    p = peaks(y, 0.5)
    return len(p) + sum(len(b) >= k for b in bouts(p, int(EDGE_GAP_S * FS)))


def edge_gyro_count(y, gy, k):
    p, g = peaks(y, 0.5), step_times(gy)
    known = list(p)
    for b in bouts(p, int(EDGE_GAP_S * FS)):
        if len(b) < k:
            continue
        gap = np.median(np.diff(b)) if len(b) > 1 else 30
        for c in ([x for x in g if 0.5 * gap <= b[0] - x <= 1.5 * gap], [x for x in g if 0.5 * gap <= x - b[-1] <= 1.5 * gap]):
            c = [x for x in c if all(abs(x - q) >= PEAK_GAP_S * FS for q in known)]   # not a step already counted
            if c:
                known.append(c[0])
    return len(known)


RULES = {f"(a) peaks >= {h}": (lambda y, tr, h=h: len(peaks(y, h))) for h in THRESHOLDS}
RULES["(b) round(sum)"] = lambda y, tr: int(np.floor(y.sum() + 0.5))
RULES.update({f"(c) edge, bouts >= {k}": (lambda y, tr, k=k: edge_count(y, k)) for k in MIN_BOUTS})
RULES.update({f"(c) edge-gyro, bouts >= {k} (offline only)": (lambda y, tr, k=k: edge_gyro_count(y, tr.gy, k)) for k in MIN_BOUTS})
OFFLINE_ONLY = [r for r in RULES if r.startswith("(c) edge-gyro, ")]
# the board-ready form of (c) edge-gyro: gy only in a window around each edge (counting.count_steps)
RULES["(c) edge-gyro-windowed (board form)"] = lambda y, tr: len(count_steps(y, tr.gy))


def folds(trials, seed):
    """FOLDS folds of whole trials; zero-step trials go to different folds."""
    rng = random.Random(seed)
    zero = sorted([t for t in trials if t.counted == 0], key=lambda t: t.name)
    walk = sorted([t for t in trials if t.counted > 0], key=lambda t: t.name)
    rng.shuffle(zero)
    rng.shuffle(walk)
    out = [[] for _ in range(FOLDS)]
    for i, t in enumerate(zero + walk):
        out[i % FOLDS].append(t)
    return out


def retrain(train_set, seed):
    """The committed run's procedure, except the kept epoch is picked on its own training trials."""
    args = Namespace(lr=3e-3, ft_lr=3e-4, batch=8, seed=seed, log_every=10 ** 9)
    per_epoch = -(-len(train_set) // args.batch)
    args.pretrain_epochs, args.epochs, args.patience = -(-90 // per_epoch), -(-2400 // per_epoch), -(-600 // per_epoch)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = det.pick_device("auto")
    mean, std = det.norm_stats(train_set)
    model = StepCNN(in_channels=len(CHANNELS)).to(device)
    det.train(model, train_set, train_set, mean, std, device, args, lambda *_: None)
    return model, mean, std, device


def main():
    trials, _ = load_trials(REPO / "data" / "raw")
    train_set, val, _, _ = det.split(trials, 0.2, 0)
    print(f"{len(train_set)} training trials (validation and Session C not used): "
          + ", ".join(t.name for t in train_set), flush=True)
    rows = []
    for seed in SEEDS:
        for i, held in enumerate(folds(train_set, seed)):
            rest = [t for t in train_set if t not in held]
            model, mean, std, device = retrain(rest, seed)
            for tr, y in zip(held, det.predict(model, held, mean, std, device)):
                row = {"seed": seed, "fold": i, "trial": tr.name, "counted": tr.counted}
                row.update({r: f(y, tr) - tr.counted for r, f in RULES.items()})
                rows.append(row)
            print(f"  seed {seed} fold {i + 1}/{FOLDS}: held out {', '.join(t.name for t in held)}", flush=True)
    df = pd.DataFrame(rows)

    table = pd.DataFrame({
        "rule": list(RULES),
        "mean abs error (steps)": [df[r].abs().mean() for r in RULES],
        "bias (steps)": [df[r].mean() for r in RULES],
        "exact trials %": [100 * (df[r] == 0).mean() for r in RULES],
    })
    for seed in SEEDS:
        table[f"MAE seed {seed}"] = [df[df.seed == seed][r].abs().mean() for r in RULES]
    print("\n" + table.round(2).to_string(index=False))

    # Committed model on its own training trials, for reference only (unfair, see the top)
    ck = torch.load(REPO / "models" / "step_detector" / "real" / "model.pt", map_location="cpu")
    committed = StepCNN(in_channels=len(CHANNELS))
    committed.load_state_dict(ck["state_dict"])
    norm = det.norm_stats(train_set)
    own = det.predict(committed, train_set, *norm, torch.device("cpu"))
    table["committed model, own training trials"] = [
        np.mean([abs(RULES[r](y, tr) - tr.counted) for y, tr in zip(own, train_set)]) for r in RULES]
    print("\n" + table.round(2).to_string(index=False))

    # Best setting of each family; the chosen rule is the best one the board can run.
    # Ties go to the earlier (simpler) entry.
    fam = table["rule"].str.extract(r"^(\(\w\) [\w-]+)")[0]
    best = table.loc[table.groupby(fam, sort=False)["mean abs error (steps)"].idxmin()]
    best = best.sort_values("mean abs error (steps)", kind="stable")
    live = best[~best["rule"].isin(OFFLINE_ONLY)]
    winner = live.iloc[0]["rule"]
    print(f"\nbest of each family:\n{best.round(2).to_string(index=False)}\n\nchosen: {winner}")

    per_trial = df.groupby("trial", sort=False)[[r for r in best["rule"]]].apply(lambda d: d.abs().mean())
    md = lambda d: "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]  # noqa: E731
                             + ["| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |"
                                for r in d.itertuples(index=False)])
    OUT.write_text(
        "# Choosing the CNN's counting rule (training trials only)\n\n"
        "Written by `analysis/counting_rule.py`. The CNN was retrained "
        f"{FOLDS * len(SEEDS)} times inside the {len(train_set)} training trials ({FOLDS} folds x "
        f"{len(SEEDS)} seeds), each time without 2 of them, with the committed run's settings. Each "
        "rule was scored on the trials each model never saw. Each model picked its kept epoch on its own "
        "training trials. The validation trials and Session C were not used.\n\n"
        "Scoring the committed model on its own training trials (last column) would be unfair. There, the "
        "count loss fitted sum = counted, so rounding the sum is exact on every trial by construction. "
        "On trials the model hadn't seen, it is the worst rule.\n\n"
        "Rules marked offline-only run the gyro stride detector on the whole trial, which the board doesn't "
        "have. The windowed form (`counting.count_steps`) runs it only on a window around each bout edge, "
        "which the board keeps in memory, so offline and live compute the same thing.\n\n"
        "## All rules (error = predicted - counted, over "
        f"{len(df)} held-out trial predictions)\n\n" + md(table.round(2)) + "\n\n"
        "## Best setting of each rule, mean abs error per held-out trial\n\n"
        + md(per_trial.reset_index()) + f"\n\n**Chosen (best that runs on the board): {winner}.** "
        "Frozen in `training/step_detector/counting.py` (`count_steps`) and `board/python/live_steps.py`.\n")
    print(f"saved {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
