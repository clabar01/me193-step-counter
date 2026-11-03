"""Step-length MLP and baselines, trained only on known trial distances.

    python training/step_length/train.py                        # real Session B distance trials
    python training/step_length/train.py --leg-length 0.9       # sets the pendulum's L (see below)

Each distance trial is split into steps (features.py). Every method predicts
one length per step; a trial's predicted distance is the sum over its steps.
There are no per-step labels, only each trial's tape-measured distance.

Methods (all fitted with the same per-trial loss, ((sum - distance) / distance)^2):
  fixed      every step the same length K
  weinberg   K * (accel range)^0.25 per step
  pendulum   K * 2 L sin(swing / 2) per step. K is fitted, so L only changes
             how K reads, not the predictions; without --leg-length, L = 1.
  mlp        5 features -> 16 -> 16 -> 1 (ReLU, softplus so lengths are > 0)

Evaluation is leave-one-trial-out: each trial is predicted by every method
fitted on the other trials only (the MLP's z-scoring included). The final
model and stats saved to models/step_length/ are fitted on all trials.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from features import FEATURES, distance_trials  # noqa: E402
from plots import error_by_speed, features_vs_speed  # noqa: E402

SPEEDS = ("slow", "normal", "fast", "jog")
METHODS = ("fixed", "weinberg", "pendulum", "mlp")
LABELS = {"fixed": "fixed mean step", "weinberg": "Weinberg", "pendulum": "pendulum", "mlp": "MLP"}
HIDDEN = 16
EPOCHS = 1500               # full batch: one update per epoch
LR = 1e-2
WEIGHT_DECAY = 1e-3


class StepLengthMLP(nn.Module):
    def __init__(self, n_in=len(FEATURES), hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        """(steps, n_in) z-scored features -> (steps,) step lengths in m."""
        return nn.functional.softplus(self.net(x)).squeeze(-1)


def trial_loss(pred_sums, distances):
    """Mean over trials of the squared relative distance error."""
    return (((pred_sums - distances) / distances) ** 2).mean()


def fit_scale(per_step, distances):
    """K minimizing sum_i ((K * S_i - D_i) / D_i)^2, S_i = trial i's summed per-step term.
    Closed form: K = sum(S/D) / sum((S/D)^2)."""
    r = np.array([s.sum() for s in per_step]) / np.asarray(distances)
    return float(r.sum() / (r ** 2).sum())


def baseline_terms(method, feats, leg_length):
    """Per-step term each baseline scales by K."""
    if method == "fixed":
        return np.ones(len(feats))
    if method == "weinberg":
        return feats[:, FEATURES.index("acc_range_g")] ** 0.25
    if method == "pendulum":
        return 2 * leg_length * np.sin(np.radians(feats[:, FEATURES.index("swing_deg")]) / 2)
    raise ValueError(method)


def fit_mlp(feats, distances, seed, on_update=None):
    """Train on whole trials (feats: list of (steps, 5) arrays). Returns (model, mean, std)."""
    allf = np.concatenate(feats)
    mean, std = allf.mean(axis=0), allf.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    xs = [torch.tensor((f - mean) / std, dtype=torch.float32) for f in feats]
    d = torch.tensor(distances, dtype=torch.float32)
    torch.manual_seed(seed)
    model = StepLengthMLP(n_in=feats[0].shape[1])
    # Start at the fixed-mean answer: softplus(bias) = average step length, so the
    # MLP only has to learn how steps differ from average.
    avg = float(d.sum() / sum(len(f) for f in feats))
    with torch.no_grad():
        model.net[-1].weight.mul_(0.1)
        model.net[-1].bias.fill_(float(np.log(np.expm1(avg))))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for step in range(EPOCHS):
        sums = torch.stack([model(x).sum() for x in xs])
        loss = trial_loss(sums, d)
        opt.zero_grad()
        loss.backward()
        if on_update:                    # snapshots for animation/record_training.py
            on_update(step, model, loss.item(), mean, std)
        opt.step()
    return model, mean, std


def predict_mlp(model, mean, std, f):
    with torch.no_grad():
        return model(torch.tensor((f - mean) / std, dtype=torch.float32)).numpy()


def leave_one_out(data, leg_length, seed):
    rows = []
    for i, (tr, f, _) in enumerate(data):
        rest = [x for j, x in enumerate(data) if j != i]
        rest_f = [x[1] for x in rest]
        rest_d = [x[0].meta["distance_m"] for x in rest]
        dist = tr.meta["distance_m"]
        row = {"trial": tr.name, "speed": tr.meta.get("speed"), "distance_m": dist,
               "counted": tr.meta.get("counted_steps"), "steps": len(f)}
        for m in METHODS[:-1]:
            k = fit_scale([baseline_terms(m, x, leg_length) for x in rest_f], rest_d)
            row[m] = float(k * baseline_terms(m, f, leg_length).sum())
        model, mean, std = fit_mlp(rest_f, rest_d, seed)
        row["mlp"] = float(predict_mlp(model, mean, std, f).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    for m in METHODS:
        df[f"{m}_err%"] = 100 * (df[m] - df["distance_m"]) / df["distance_m"]
    return df


def summary(df):
    rows = {}
    for speed in [s for s in SPEEDS if s in set(df["speed"])] + ["all"]:
        d = df if speed == "all" else df[df["speed"] == speed]
        rows[f"{speed} ({len(d)})"] = {LABELS[m]: round(d[f"{m}_err%"].abs().mean(), 1) for m in METHODS}
    return pd.DataFrame(rows).T


def markdown(df):
    cells = [[("" if pd.isna(v) else (f"{v:.1f}" if isinstance(v, float) else str(v))) for v in r]
             for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells])


def main():
    ap = argparse.ArgumentParser(description="Step-length MLP and baselines on distance trials.")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--leg-length", type=float, help="leg length, m (pendulum only; default 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", help="output name (default: sim if --data-dir is under data/sim, else real)")
    args = ap.parse_args()
    data_dir = args.data_dir.resolve()
    is_sim = data_dir.is_relative_to(REPO) and "sim" in data_dir.relative_to(REPO).parts
    tag = args.tag or ("sim" if is_sim else "real")
    leg = args.leg_length or 1.0
    out, figs = REPO / "models" / "step_length" / tag, REPO / "docs" / "figures" / "step_length" / tag
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    data, skipped = distance_trials(data_dir)
    data = [x for x in data if len(x[1])]
    if len(data) < 3:
        raise SystemExit(f"need at least 3 distance trials with steps, found {len(data)}")
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    log(f"data: {data_dir.name}/ ({'SIMULATED' if is_sim else 'real'}), {len(data)} distance trials, "
        f"skipped {len(skipped)} bench_*; leg length {'not given, L = 1 m' if not args.leg_length else leg}")
    for tr, f, _ in data:
        log(f"  {tr.name}: {tr.meta['speed']}, {tr.meta['distance_m']:g} m, {len(f)} steps "
            f"(counted {tr.meta.get('counted_steps')}), {tr.meta['distance_m'] / len(f):.3f} m/step")

    df = leave_one_out(data, leg, args.seed)
    table = summary(df)
    log("\nleave-one-trial-out, mean |distance error| %:\n" + table.to_string())
    per_trial = df[["trial", "speed", "distance_m", "steps"] + [f"{m}_err%" for m in METHODS]]
    log("\nper trial, distance error %:\n" + per_trial.round(1).to_string(index=False))

    # seed check: does the MLP's answer depend on its random start?
    seeds = [leave_one_out(data, leg, s)["mlp_err%"].abs().mean() for s in range(1, 4)]
    log(f"\nMLP mean |error| with other seeds: {', '.join(f'{v:.1f}%' for v in seeds)}")

    # final fits on all trials
    feats, dists = [x[1] for x in data], [x[0].meta["distance_m"] for x in data]
    ks = {m: fit_scale([baseline_terms(m, f, leg) for f in feats], dists) for m in METHODS[:-1]}
    model, mean, std = fit_mlp(feats, dists, args.seed)
    torch.save({"state_dict": model.state_dict(), "features": FEATURES, "hidden": HIDDEN}, out / "model.pt")
    (out / "norm.json").write_text(json.dumps({"features": FEATURES, "mean": np.round(mean, 6).tolist(),
                                               "std": np.round(std, 6).tolist(),
                                               "from": f"{len(data)} distance trials"}, indent=1) + "\n")
    (out / "baselines.json").write_text(json.dumps({
        "fixed_step_m": round(ks["fixed"], 4), "weinberg_K": round(ks["weinberg"], 4),
        "pendulum_scale": round(ks["pendulum"], 4), "pendulum_leg_length_m": leg}, indent=1) + "\n")
    df.round(3).to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(
        f"# Step length results ({tag})\n\nLeave-one-trial-out over {len(data)} distance trials "
        f"({'SIMULATED' if is_sim else 'real'}). Each trial is predicted by methods fitted on the "
        f"other {len(data) - 1}.\n\n## Mean |distance error| %, by speed (number of trials)\n\n"
        + markdown(table.reset_index(names="speed")) + "\n\n## Per trial, distance error %\n\n"
        + markdown(per_trial) + "\n\nFitted on all trials: " + ", ".join(f"{k} {v:.4f}" for k, v in ks.items())
        + "\n")
    features_vs_speed(data, figs / "features_vs_speed.png")
    error_by_speed(df, METHODS, LABELS, SPEEDS, figs / "error_by_speed.png",
                   f"Distance error, leave-one-trial-out ({len(data)} {'SIMULATED' if is_sim else 'real'} trials)")
    log(f"\nsaved {out.relative_to(REPO)}/ and {figs.relative_to(REPO)}/")
    (out / "train_log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
