"""Uncertainty analysis of the step counter and distance estimate.

    python analysis/evaluate.py               # tonight: the validation trials stand in for the test set
    python analysis/evaluate.py --test C      # with Session C recorded: the real held-out test set

Writes analysis/results.md (plain language, with the numbers) and figures in
docs/figures/evaluation/<val|C>/.

1. Step count: per test trial, counted vs predicted by the CNN (committed
   real-data model, counted with counting.count_steps: peaks + gyro-checked
   edge steps, chosen on the training trials by analysis/counting_rule.py; the
   old peaks-only rule is shown for reference) and by the gyro stride detector; mean absolute error, SD,
   bias and 95% bootstrap confidence intervals, overall and by speed; a
   predicted-vs-counted scatter.
2. Distance: % error per distance trial for the step-length MLP and the
   baselines (fixed step, Weinberg, pendulum), and end-to-end (MLP on the CNN's
   steps, as on the board), by speed with confidence intervals.
     --test val: every A/B distance trial, each predicted by models fitted on
                 the other distance trials (leave-one-trial-out), since the
                 step-length model was trained on all of them.
     --test C:   Session C trials that have a distance, predicted by models
                 fitted on all A/B distance trials.
3. Ablation (same splits and seed): step detector retrained with the
   ultrasonic channels held at "no echo" (IMU only) vs the committed IMU +
   ultrasonic model; step-length MLP with and without the two ultrasonic features.
4. How step-count errors carry into distance errors.
5. Limitations.

Confidence intervals are 95% percentile bootstraps over trials (10,000
resamples). With 2-4 trials per speed they are wide: they say how little a
handful of trials can pin down, not that the method is bad.
"""

import argparse
import importlib.util
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
DET_DIR = REPO / "training" / "step_detector"
LEN_DIR = REPO / "training" / "step_length"
sys.path.insert(0, str(DET_DIR))
import train as det  # noqa: E402
from counting import count_steps, peak_steps, sum_count  # noqa: E402
from data import CHANNELS, US_CLIP_MM, load_trials, preprocess  # noqa: E402
from model import StepCNN  # noqa: E402
from strides import step_times  # noqa: E402


def load_step_length():
    """training/step_length's train.py and features.py (their names clash with step_detector's)."""
    clashing = {k: sys.modules.pop(k) for k in ("train", "plots", "features") if k in sys.modules}
    sys.path.insert(0, str(LEN_DIR))
    mods = {}
    for name in ("features", "train"):
        spec = importlib.util.spec_from_file_location(f"step_length_{name}", LEN_DIR / f"{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        sys.modules[name] = mods[name] if name == "features" else sys.modules.get(name)
        spec.loader.exec_module(mods[name])
    sys.path.remove(str(LEN_DIR))
    for k in ("train", "plots", "features"):
        sys.modules.pop(k, None)
    sys.modules.update(clashing)
    return mods["features"], mods["train"]


slf, slt = load_step_length()

SPEEDS = ("slow", "normal", "fast", "jog")
B = 10_000
FIG = None                                   # set in main()
INK, MUTED, GRID, TEXT = "#1d1d1b", "#52514e", "#e4e3df", "#0b0b0b"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")   # categorical slots 1-5, fixed per method
DIST_METHODS = ("fixed", "weinberg", "pendulum", "mlp", "mlp_cnn")
DIST_LABELS = {"fixed": "fixed step", "weinberg": "Weinberg", "pendulum": "pendulum", "mlp": "MLP",
               "mlp_cnn": "MLP on CNN steps (board)"}


# ---------------- statistics ----------------
def boot_ci(v, stat=np.mean, seed=0):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = stat(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return tuple(np.percentile(s, [2.5, 97.5]))


def fmt_ci(ci, nd=2):
    return "n/a" if np.isnan(ci[0]) else f"{ci[0]:.{nd}f} to {ci[1]:.{nd}f}"


def group(row):
    return "standing (0 steps)" if row["counted"] == 0 else row["speed"]


def groups_of(df):
    present = list(dict.fromkeys(df["group"]))
    order = [g for g in SPEEDS if g in present] + [g for g in present if g not in SPEEDS]
    return [(g, df[df["group"] == g]) for g in order] + [("all", df)]


def md(df):
    cells = [["" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v))
              for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells])


def style(ax, ylabel=None, xlabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


# ---------------- models ----------------
def load_detector():
    ck = torch.load(REPO / "models" / "step_detector" / "real" / "model.pt", map_location="cpu")
    m = StepCNN()
    m.load_state_dict(ck["state_dict"])
    norm = json.loads((REPO / "models" / "step_detector" / "real" / "norm.json").read_text())
    return m.eval(), np.array(norm["mean"]), np.array(norm["std"])


def cnn_output(model, mean, std, x):
    with torch.no_grad():
        z = torch.tensor(((x - mean) / std).T[None], dtype=torch.float32)
        return model(z)[0].numpy()


def imu_only(x):
    """Ultrasonic channels held at 'no echo': the model gets no ultrasonic information."""
    x = x.copy()
    x[:, CHANNELS.index("us_mm")] = US_CLIP_MM
    x[:, CHANNELS.index("us_hit")] = 0
    return x


# ---------------- 1. step count ----------------
def step_table(test, model, mean, std):
    rows = []
    for tr in test:
        y = cnn_output(model, mean, std, tr.x)
        rows.append({"trial": tr.name, "speed": tr.meta.get("speed"), "counted": tr.counted,
                     "cnn": len(count_steps(y, tr.gy)), "cnn_peaks": len(peak_steps(y)),
                     "cnn_sum": round(sum_count(y), 1), "gyro": len(step_times(tr.gy))})
    df = pd.DataFrame(rows)
    df["group"] = df.apply(group, axis=1)
    return df


def step_summary(df, methods):
    rows = []
    for g, d in groups_of(df):
        for m, label in methods:
            e = (d[m] - d["counted"]).to_numpy(float)
            walk = d[d["counted"] > 0]
            rows.append({
                "speed": g, "method": label, "trials": len(d), "steps counted": int(d["counted"].sum()),
                "mean abs error (steps)": float(np.abs(e).mean()),
                "95% CI": fmt_ci(boot_ci(np.abs(e))),
                "bias (steps)": float(e.mean()), "bias 95% CI": fmt_ci(boot_ci(e)),
                "SD (steps)": float(e.std(ddof=1)) if len(e) > 1 else np.nan,
                "total error %": float(100 * (walk[m].sum() - walk["counted"].sum()) / walk["counted"].sum())
                if walk["counted"].sum() else np.nan,
            })
    return pd.DataFrame(rows)


def step_scatter(df, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 6), layout="constrained")
    hi = max(df[["counted", "cnn", "gyro"]].max()) + 3
    ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1.2, linestyle=(0, (4, 4)), label="predicted = counted")
    ax.scatter(df["counted"], df["gyro"], s=70, color=SERIES[1], label="gyro detector", zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.scatter(df["counted"], df["cnn"], s=70, color=SERIES[0], label="CNN", zorder=3, edgecolor="white", linewidth=1.5)
    for _, r in df.iterrows():
        ax.annotate(r["trial"], (r["counted"], r["cnn"]), textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color=MUTED)
    ax.set_xlim(-1, hi)
    ax.set_ylim(-1, hi)
    style(ax, "predicted steps", "steps counted by hand")
    ax.legend(frameon=False, labelcolor=MUTED, loc="upper left")
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------- 2. distance ----------------
def cnn_steps(model, mean, std, x):
    """The board's steps: the counting rule on the CNN output."""
    return count_steps(cnn_output(model, mean, std, x), x[:, CHANNELS.index("gy")])


def fit_length_models(train_items, cols, seed=0):
    # contiguous like the original arrays: a column-sliced copy changes the order of floating-point
    # sums, and over 1,500 updates that moves the results by ~0.2 percentage points per trial
    feats = [np.ascontiguousarray(it["f"][:, cols]) for it in train_items]
    dists = [it["dist"] for it in train_items]
    ks = {m: slt.fit_scale([slt.baseline_terms(m, it["f"], 1.0) for it in train_items], dists)
          for m in ("fixed", "weinberg", "pendulum")}
    model, mu, sd = slt.fit_mlp(feats, dists, seed)
    return ks, (model, mu, sd)


def predict_distance(item, ks, mlp, cols):
    model, mu, sd = mlp
    out = {m: float(ks[m] * slt.baseline_terms(m, item["f"], 1.0).sum()) for m in ks}
    out["mlp"] = float(slt.predict_mlp(model, mu, sd, np.ascontiguousarray(item["f"][:, cols])).sum())
    out["mlp_cnn"] = (float(slt.predict_mlp(model, mu, sd, np.ascontiguousarray(item["f_cnn"][:, cols])).sum())
                      if len(item["f_cnn"]) else np.nan)
    return out


def distance_items(trials, model, mean, std):
    items = []
    for tr in trials:
        if tr.meta.get("type") != "distance" or not tr.meta.get("distance_m"):
            continue
        f, steps = slf.step_features(tr.x)
        if not len(f):
            continue
        cs = cnn_steps(model, mean, std, tr.x)
        f_cnn, _ = slf.step_features(tr.x, cs)
        items.append({"trial": tr, "f": f, "steps": steps, "f_cnn": f_cnn, "cnn_steps": len(cs),
                      "dist": tr.meta["distance_m"]})
    return items


def distance_table(items, test_items, mode, cols=list(range(5)), seed=0):
    """mode 'loo': each of items predicted by models fitted on the others; 'fixed': fit on items, predict test_items."""
    rows = []
    targets = items if mode == "loo" else test_items
    fixed_fit = None if mode == "loo" else fit_length_models(items, cols, seed)
    for it in targets:
        ks, mlp = fit_length_models([o for o in items if o is not it], cols, seed) if mode == "loo" else fixed_fit
        pred = predict_distance(it, ks, mlp, cols)
        tr = it["trial"]
        row = {"trial": tr.name, "speed": tr.meta.get("speed"), "distance_m": it["dist"], "counted": tr.counted,
               "gyro steps": len(it["steps"]), "CNN steps": it["cnn_steps"]}
        for m in DIST_METHODS:
            row[f"{m}_m"] = pred[m]
            row[f"{m}_err%"] = 100 * (pred[m] - it["dist"]) / it["dist"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["group"] = df["speed"]
    return df


def distance_summary(df):
    rows = []
    for g, d in groups_of(df):
        for m in DIST_METHODS:
            e = d[f"{m}_err%"].dropna().to_numpy(float)
            rows.append({"speed": g, "method": DIST_LABELS[m], "trials": len(e),
                         "mean abs error %": float(np.abs(e).mean()) if len(e) else np.nan,
                         "95% CI": fmt_ci(boot_ci(np.abs(e)), 1),
                         "bias %": float(e.mean()) if len(e) else np.nan, "bias 95% CI": fmt_ci(boot_ci(e), 1)})
    return pd.DataFrame(rows)


def distance_figure(df, path, title):
    groups = [g for g, _ in groups_of(df)]
    width = 0.8 / len(DIST_METHODS)
    fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
    for k, m in enumerate(DIST_METHODS):
        for gi, g in enumerate(groups):
            d = df if g == "all" else df[df["group"] == g]
            e = np.abs(d[f"{m}_err%"].dropna().to_numpy(float))
            if not len(e):
                continue
            x = gi + (k - (len(DIST_METHODS) - 1) / 2) * width
            lo, hi = boot_ci(e)
            ax.bar(x, e.mean(), width * 0.9, color=SERIES[k], label=DIST_LABELS[m] if gi == 0 else None)
            if not np.isnan(lo):
                ax.errorbar(x, e.mean(), yerr=[[e.mean() - lo], [hi - e.mean()]], color=INK, capsize=3, linewidth=1.2)
            ax.text(x, 0.15, f"{e.mean():.1f}", ha="center", va="bottom", fontsize=8, color="white", rotation=90)
    counts = [len(df) if g == "all" else int((df["group"] == g).sum()) for g in groups]
    ax.set_xticks(range(len(groups)), [f"{g}\n({n} trial{'s' if n != 1 else ''})" for g, n in zip(groups, counts)])
    style(ax, "mean |distance error| (%), 95% CI")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, labelcolor=MUTED, ncols=len(DIST_METHODS), loc="upper left", fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def count_vs_distance_figure(df, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    for col, n_col, m, color, label in (("gyro steps", "mlp", "mlp", SERIES[3], "MLP on gyro steps"),
                                        ("CNN steps", "mlp_cnn", "mlp_cnn", SERIES[4], "MLP on CNN steps (board)")):
        ce = 100 * (df[col] - df["counted"]) / df["counted"]
        ax.scatter(ce, df[f"{m}_err%"], s=70, color=color, label=label, edgecolor="white", linewidth=1.5, zorder=3)
        for (_, r), c in zip(df.iterrows(), ce):
            ax.annotate(r["trial"], (c, r[f"{m}_err%"]), textcoords="offset points", xytext=(6, 4), fontsize=8, color=MUTED)
    lim = max(8, np.nanmax(np.abs(df[[f"mlp_err%", f"mlp_cnn_err%"]].to_numpy())) + 2)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linestyle=(0, (4, 4)), linewidth=1.2, label="distance error = count error")
    ax.axhline(0, color=GRID)
    ax.axvline(0, color=GRID)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    style(ax, "distance error (%)", "step count error vs counted (%)")
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=8, loc="upper left")
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------- 3. ablation ----------------
ABL_SEEDS = (0, 1, 2)


def retrain_detector(train_set, val, seed, transform):
    """The committed model's exact procedure (device, settings, split), with a given seed and inputs.
    Seed 0 with the inputs unchanged reproduces the committed model."""
    tr_set = [_copy_with(t, transform(t.x)) for t in train_set]
    va_set = [_copy_with(t, transform(t.x)) for t in val]
    args = Namespace(lr=3e-3, ft_lr=3e-4, batch=8, seed=seed, log_every=10 ** 9)
    per_epoch = -(-len(tr_set) // args.batch)
    args.pretrain_epochs, args.epochs, args.patience = -(-90 // per_epoch), -(-2400 // per_epoch), -(-600 // per_epoch)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = det.pick_device("auto")
    mean, std = det.norm_stats(tr_set)
    model = StepCNN(in_channels=len(CHANNELS)).to(device)
    det.train(model, tr_set, va_set, mean, std, device, args, lambda *_: None)
    return model.eval().cpu(), mean, std


def _copy_with(tr, x):
    from copy import copy
    t = copy(tr)
    t.x = x.astype(np.float32)
    return t


# ---------------- main ----------------
def main():
    global FIG
    ap = argparse.ArgumentParser(description="Uncertainty analysis: step count, distance, ablation.")
    ap.add_argument("--test", choices=("val", "C"), default="val",
                    help="val: validation trials stand in for the test set; C: Session C (the real test set)")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--skip-ablation", action="store_true", help="skip retraining (faster)")
    args = ap.parse_args()
    FIG = REPO / "docs" / "figures" / "evaluation" / args.test
    log = lambda m: print(m, flush=True)  # noqa: E731

    trials, skipped = load_trials(args.data_dir)
    train_set, val, test_c, _ = det.split(trials, 0.2, 0)
    test = val if args.test == "val" else test_c
    if not test:
        raise SystemExit("no Session C trials in data/raw yet: record them, pull them, then run --test C")
    FIG.mkdir(parents=True, exist_ok=True)
    label = ("VALIDATION trials standing in for the test set (they also chose the kept training epoch, "
             "so these numbers are slightly optimistic)" if args.test == "val"
             else "Session C, the held-out test set (never used in training or model selection)")
    log(f"test set: {len(test)} trials ({args.test})")
    model, mean, std = load_detector()

    # 1. step count
    steps = step_table(test, model, mean, std)
    methods = [("cnn", "CNN"), ("cnn_peaks", "CNN, old rule (peaks only)"), ("gyro", "gyro detector")]
    step_sum = step_summary(steps, methods)
    step_scatter(steps, FIG / "step_scatter.png", f"Predicted vs counted steps ({len(steps)} {args.test} trials)")
    log(step_sum[step_sum["speed"] == "all"].to_string(index=False))

    # 2. distance
    ab_trials = [t for t in trials if t.session in det.TRAIN_SESSIONS]
    ab_items = distance_items(ab_trials, model, mean, std)
    if args.test == "val":
        dist = distance_table(ab_items, None, "loo")
        dist_note = (f"all {len(dist)} A/B distance trials, each predicted by models fitted on the other "
                     f"{len(dist) - 1} (leave-one-trial-out)")
    else:
        test_items = distance_items(test, model, mean, std)
        dist = distance_table(ab_items, test_items, "fixed")
        dist_note = f"{len(dist)} Session C distance trials, models fitted on all {len(ab_items)} A/B distance trials"
    dist_sum = distance_summary(dist) if len(dist) else pd.DataFrame()
    if len(dist):
        distance_figure(dist, FIG / "distance_by_speed.png", f"Distance error by speed ({dist_note.split(',')[0]})")
        count_vs_distance_figure(dist, FIG / "count_vs_distance.png", "Step-count error vs distance error")
        log(dist_sum[dist_sum["speed"] == "all"].to_string(index=False))

    # 3. ablation: both variants retrained the same way, with 3 seeds each
    abl_rows, abl_paired, len_paired, abl_len_seeds = [], None, None, {}
    if not args.skip_ablation:
        per_variant = {}
        for name, transform in (("IMU + ultrasonic", lambda x: x), ("IMU only", imu_only)):
            runs = []
            for seed in ABL_SEEDS:
                log(f"  retraining the step detector, {name}, seed {seed}...")
                m2, mean2, std2 = retrain_detector(train_set, val, seed, transform)
                runs.append(step_table([_copy_with(t, transform(t.x)) for t in test], m2, mean2, std2))
            per_variant[name] = runs
            maes = [float(np.abs(r["cnn"] - r["counted"]).mean()) for r in runs]
            tot = [float(100 * (r["cnn"].sum() - r["counted"].sum()) / max(r["counted"].sum(), 1)) for r in runs]
            abl_rows.append({"model": "step detector", "inputs": name,
                             "mean abs error per seed": ", ".join(f"{v:.2f}" for v in maes) + " steps",
                             "mean over seeds": f"{np.mean(maes):.2f} steps (total {np.mean(tot):+.1f}%)"})
        same = (per_variant["IMU + ultrasonic"][0]["cnn"].to_numpy() == steps["cnn"].to_numpy()).all()
        log(f"  seed 0 with IMU + ultrasonic reproduces the committed model's counts: {same}")
        steps["cnn_imu_only"] = per_variant["IMU only"][0]["cnn"].to_numpy()
        err = {k: np.mean([np.abs(r["cnn"] - r["counted"]).to_numpy(float) for r in v], axis=0)
               for k, v in per_variant.items()}
        abl_paired = boot_ci(err["IMU only"] - err["IMU + ultrasonic"])
        if len(dist):
            test_items = None if args.test == "val" else distance_items(test, model, mean, std)
            per_len = {}
            for cols, name in ((list(range(5)), "all 5 features"), ([0, 1, 2], "IMU features only")):
                runs = [distance_table(ab_items, test_items, "loo" if args.test == "val" else "fixed", cols, seed)
                        for seed in ABL_SEEDS]
                per_len[name] = np.mean([np.abs(r["mlp_err%"].to_numpy(float)) for r in runs], axis=0)
                maes = [float(np.abs(r["mlp_err%"]).mean()) for r in runs]
                abl_len_seeds[name] = maes
                abl_rows.append({"model": "step-length MLP", "inputs": name,
                                 "mean abs error per seed": ", ".join(f"{v:.1f}" for v in maes) + "% distance",
                                 "mean over seeds": f"{np.mean(maes):.1f}%"})
            len_paired = boot_ci(per_len["IMU features only"] - per_len["all 5 features"])
    ablation = pd.DataFrame(abl_rows)
    log(ablation.to_string(index=False))

    # bench_05: the turning limitation
    turn = None
    bench = args.data_dir / "bench_05.csv"
    if bench.exists():
        meta = json.loads(bench.with_suffix(".json").read_text())
        x, _ = preprocess(pd.read_csv(bench), int(meta.get("us_lag_samples", 1)))
        turn = (meta["counted_steps"], len(cnn_steps(model, mean, std, x)), len(step_times(x[:, CHANNELS.index("gy")])))

    # ---------------- write results.md ----------------
    sa = step_sum[step_sum["speed"] == "all"].set_index("method")
    out = [f"# Results: uncertainty analysis\n",
           f"Made by `python analysis/evaluate.py --test {args.test}`. **Test set: {label}.**",
           f"Trials: {', '.join(steps['trial'])}. Confidence intervals (CI) are 95% bootstrap intervals over "
           "trials: with 2-4 trials per speed they are wide, and say how much a handful of trials can't pin down.\n",
           "## 1. Step count\n",
           f"Over all {len(steps)} test trials ({int(steps['counted'].sum())} steps counted by hand), the CNN was off by "
           f"**{sa.loc['CNN', 'mean abs error (steps)']:.2f} steps per trial** on average "
           f"(95% CI {sa.loc['CNN', '95% CI']}), and the gyro stride detector by "
           f"**{sa.loc['gyro detector', 'mean abs error (steps)']:.2f}** (95% CI {sa.loc['gyro detector', '95% CI']}). "
           f"Bias (average of predicted − counted): CNN {sa.loc['CNN', 'bias (steps)']:+.2f} steps, gyro "
           f"{sa.loc['gyro detector', 'bias (steps)']:+.2f}. A negative bias means steps are missed.\n",
           "The CNN's steps are counted with the rule chosen on the training trials only "
           "(`analysis/counting_rule.md`). It counts the peaks of the CNN output, plus one step at the start or "
           "end of a walk when the gyro stride detector has a step there that the CNN missed. The CNN nearly "
           "always misses the last step of a walk: its output there is ~0, not a weak peak. With the old "
           f"peaks-only rule the CNN was off by {sa.loc['CNN, old rule (peaks only)', 'mean abs error (steps)']:.2f} "
           f"steps per trial (bias {sa.loc['CNN, old rule (peaks only)', 'bias (steps)']:+.2f}). The plain sum of its "
           f"output (what the count loss trains) is off by {np.abs(steps['cnn_sum'] - steps['counted']).mean():.2f}.\n",
           ({-1: "On these straight walks the plain gyro detector counts more accurately than the CNN. ",
             0: "On these straight walks the CNN and the plain gyro detector count equally well. ",
             1: "The CNN counts more accurately than the gyro detector on these walks. "}[
               int(np.sign(round(sa.loc['gyro detector', 'mean abs error (steps)'], 6)
                           - round(sa.loc['CNN', 'mean abs error (steps)'], 6)))]
            + "The agreement is partly by construction: the edge steps come from the gyro detector. The CNN still "
              "decides every step inside a walk and whether there is a walk at all (it counts sitting down as 0, "
              "where the gyro detector counts a stride).\n"),
           "Per trial:\n", md(steps[["trial", "speed", "counted", "cnn", "cnn_peaks", "cnn_sum", "gyro"]
                                    + (["cnn_imu_only"] if "cnn_imu_only" in steps else [])].rename(columns={
               "cnn": "CNN", "cnn_peaks": "CNN, old rule", "cnn_sum": "CNN sum of output", "gyro": "gyro detector",
               "cnn_imu_only": "CNN, IMU only (ablation)"})),
           "\nBy speed:\n", md(step_sum),
           f"\n![predicted vs counted](../docs/figures/evaluation/{args.test}/step_scatter.png)\n"]
    if len(dist):
        da = dist_sum[dist_sum["speed"] == "all"].set_index("method")
        out += ["## 2. Distance\n",
                f"Data: {dist_note}. Each method predicts a length for every step; the trial's distance is their sum.\n",
                ("The step-length models never saw the trial they predict, but the CNN (used for \"MLP on CNN "
                 "steps\") was trained on some of these walks, so that column is optimistic for them.\n"
                 if args.test == "val" else ""),
                f"Mean distance error: MLP **{da.loc['MLP', 'mean abs error %']:.1f}%** "
                f"(95% CI {da.loc['MLP', '95% CI']}), Weinberg {da.loc['Weinberg', 'mean abs error %']:.1f}% "
                f"({da.loc['Weinberg', '95% CI']}), pendulum {da.loc['pendulum', 'mean abs error %']:.1f}% "
                f"({da.loc['pendulum', '95% CI']}), fixed step {da.loc['fixed step', 'mean abs error %']:.1f}% "
                f"({da.loc['fixed step', '95% CI']}). On the board the MLP runs on the CNN's steps instead of the gyro "
                f"detector's: {da.loc['MLP on CNN steps (board)', 'mean abs error %']:.1f}% "
                f"({da.loc['MLP on CNN steps (board)', '95% CI']}), bias "
                f"{da.loc['MLP on CNN steps (board)', 'bias %']:+.1f}%.\n",
                "The intervals of the MLP, Weinberg and pendulum overlap: with this many trials they can't be told "
                "apart. A fixed step length is clearly worse at slow and fast speeds, because step length changes with speed.\n",
                "Per trial (distance error %):\n",
                md(dist[["trial", "speed", "distance_m", "counted", "gyro steps", "CNN steps"]
                        + [f"{m}_err%" for m in DIST_METHODS]].rename(columns={f"{m}_err%": DIST_LABELS[m] for m in DIST_METHODS})),
                "\nBy speed:\n", md(dist_sum),
                f"\n![distance error by speed](../docs/figures/evaluation/{args.test}/distance_by_speed.png)\n"]
    out += ["## 3. Ablation: what the ultrasonic adds\n",
            "Each variant is trained exactly like the committed model (same split, settings and device), three "
            f"times with seeds {', '.join(map(str, ABL_SEEDS))}; only the inputs change. For the step detector, "
            "\"IMU only\" means the ultrasonic channels are held at \"no echo\", so it can't use them. Seed 0 with "
            "IMU + ultrasonic is the committed model. Training on so few trials is sensitive to the random start, "
            "so differences between single runs mean little; look at the spread across seeds.\n",
            md(ablation) if len(ablation) else "(skipped: run without --skip-ablation)"]
    if abl_paired is not None:
        out.append(f"\nPaired difference per test trial, averaged over the seeds (IMU only − IMU + ultrasonic): "
                   f"absolute step error 95% CI {fmt_ci(abl_paired)} steps"
                   + (f"; absolute distance error {fmt_ci(len_paired, 1)} percentage points" if len_paired else "")
                   + ". A negative value means the variant without the ultrasonic did better; an interval that "
                   "includes 0 means this data can't tell them apart.\n")
        if abl_paired[1] < 0:
            out.append("**Step detector:** on these trials it counted slightly better *without* the ultrasonic. "
                       "The seeds vary as much as the variants do (one run of each is as good as the other's best), "
                       + ("and these validation trials also chose the kept epoch, " if args.test == "val" else "")
                       + "so this is weak evidence: the ultrasonic does not clearly help counting straight walks, "
                       "where the gyro swing alone is a strong signal. Session C will say more.\n")
        elif abl_paired[0] > 0:
            out.append("**Step detector:** the ultrasonic helps counting on these trials.\n")
        else:
            out.append("**Step detector:** this data can't tell whether the ultrasonic helps counting.\n")
        if len_paired is not None:
            w, wo = (abl_len_seeds["all 5 features"], abl_len_seeds["IMU features only"])
            every = all(x < y for x, y in zip(w, wo))
            if len_paired[0] > 0:
                verdict = "the ultrasonic features help: the paired interval is above 0."
            elif len_paired[1] < 0:
                verdict = (f"the ultrasonic features made the distance clearly worse on these trials: without them "
                           f"the error was {np.mean(wo):.1f}% vs {np.mean(w):.1f}%, lower in "
                           + ("every seed" if all(y < x for x, y in zip(w, wo)) else "most seeds")
                           + f", and the whole paired interval ({fmt_ci(len_paired, 1)} points) is below 0. "
                           "The step-length MLP leans on the ultrasonic features, so a change in the echoes (clothing, "
                           "sensor aim) carries straight into the step lengths.")
            elif every and len_paired[1] > 0:
                verdict = (f"without the ultrasonic features the distance error was higher in every seed "
                           f"({np.mean(wo):.1f}% vs {np.mean(w):.1f}%), but the paired interval "
                           f"({fmt_ci(len_paired, 1)} points) just includes 0: with this few trials that is "
                           "suggestive, not proven.")
            else:
                verdict = "this data can't show that the ultrasonic features help."
            out.append("**Step length:** " + verdict + "\n")
    if len(dist):
        dd = dist.copy()
        dd["gyro count err %"] = 100 * (dd["gyro steps"] - dd["counted"]) / dd["counted"]
        dd["CNN count err %"] = 100 * (dd["CNN steps"] - dd["counted"]) / dd["counted"]
        dd["length-only err %"] = 100 * (dd["mlp_m"] * dd["counted"] / dd["gyro steps"] - dd["distance_m"]) / dd["distance_m"]
        r = np.corrcoef(dd["CNN count err %"], dd["mlp_cnn_err%"])[0, 1] if len(dd) > 2 else np.nan
        worst = dd[dd["trial"] == "B_04"].iloc[0] if (dd["trial"] == "B_04").any() else dd.loc[dd["gyro count err %"].abs().idxmax()]
        out += ["\n## 4. How step-count errors become distance errors\n",
                "Distance = number of steps × their lengths, so a missed or extra step moves the distance by about "
                "one step's share: on a 25-step walk, one extra step is +4%. The table splits each trial's distance "
                "error into the count part and what's left (\"length-only\": the prediction rescaled to the hand-counted "
                "number of steps).\n",
                md(dd[["trial", "counted", "gyro steps", "gyro count err %", "mlp_err%", "length-only err %",
                       "CNN steps", "CNN count err %", "mlp_cnn_err%"]].rename(columns={
                    "mlp_err%": "MLP distance err %", "mlp_cnn_err%": "board (CNN steps) distance err %"})),
                f"\nExample: {worst['trial']} has {int(worst['gyro steps'])} gyro steps for {int(worst['counted'])} "
                f"counted ({worst['gyro count err %']:+.1f}%), and its MLP distance error is {worst['mlp_err%']:+.1f}%; "
                f"rescaled to the right step count it would be {worst['length-only err %']:+.1f}%. "
                + (f"On the board's path (CNN steps), step-count error and distance error are correlated "
                   f"(r = {r:.2f} over {len(dd)} trials): " if not np.isnan(r) else "")
                + "a step missed by the detector costs distance even if the step lengths are right.\n",
                f"![count vs distance](../docs/figures/evaluation/{args.test}/count_vs_distance.png)\n"]
    lim = ["\n## 5. Limitations\n",
           "- **Walking only, straight paths only.** All training walks are straight hallway passes at slow, normal "
           "and fast speeds (no jogging trials were recorded). "
           + (f"In a small room with turns (`bench_05`, {turn[0]} steps counted) the CNN found only **{turn[1]}** "
              f"(the gyro detector {turn[2]}): turning twists the thigh and breaks up the swing pattern it learned. "
              if turn else "")
           + "Stairs, running and turning are not covered.",
           f"- **Small dataset.** {len(train_set)} training and {len(val)} validation trials; 8 distance trials. "
           "Per-speed results rest on 2-4 trials, hence the wide intervals; the ablation can only detect large effects.",
           "- **One person, one strap position, one hallway.** Everything was recorded by one person with the "
           "sensor just above the knee, mostly in one indoor hallway. Another body, gait, strap position or floor "
           "may need recalibration or more data.",
           "- **Hand counts as truth.** Counted steps can be off by one (the first or last step is ambiguous); "
           "distances were measured with a tape.",
           "- **Live delay.** On the board a step is counted about 1.5 s after it happens (measured 1.34-1.72 s on "
           "the board's clock): the CNN needs 1.2 s of data after each step. The public MQTT broker added another "
           "0.7-4.7 s on the evening it was measured, so the dashboard can lag 2-6 s.",
           ("- **Validation as a stand-in.** These numbers use the validation trials, which also picked the kept "
            "training epoch: expect Session C to be somewhat worse." if args.test == "val" else
            "- **Held-out test set.** Session C was never used for training or model selection, so these are honest "
            "test numbers. With 6 trials (2 per speed) the intervals are still wide."),
           ]
    out += lim
    (REPO / "analysis" / "results.md").write_text("\n".join(out) + "\n")
    log(f"\nsaved analysis/results.md and docs/figures/evaluation/{args.test}/")


if __name__ == "__main__":
    main()
