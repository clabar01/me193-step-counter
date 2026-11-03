"""Train and evaluate the CNN step detector (see docs/step_detector.md).

    python training/step_detector/train.py                          # simulated data (default)
    python training/step_detector/train.py --data-dir data/raw      # real Session A/B/C trials

Pipeline:
  1. Load every trial in --data-dir (bench_* skipped), preprocess (despike,
     line up the ultrasonic, clip it at 150 mm; data.py).
  2. Split by whole trial: sessions A and B train/validation, session C is
     only ever the test set. Other sessions are ignored.
  3. z-score each channel with the TRAINING trials' mean and std only; saved
     with the model so live data is scaled the same way.
  4. Warm start: 15 epochs on pseudo-labels from the gyro stride detector.
  5. Fine-tune with the count loss (sum of outputs vs counted_steps) plus a
     peak penalty; early stopping on the validation count error.
  6. Count steps (peaks and plain sum), compare with the gyro detector alone,
     and on simulated data score step timing against the true step times.

Outputs: model + normalization + results in models/step_detector/<tag>/,
figures in docs/figures/step_detector/<tag>/; <tag> is "sim" or "real".
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from counting import event_f1, peak_steps, sum_count  # noqa: E402
from data import CHANNELS, US_CLIP_MM, load_trials, true_step_samples  # noqa: E402
from losses import count_loss, peak_penalty, pseudo_label_loss, pseudo_targets  # noqa: E402
from model import StepCNN, n_parameters, receptive_field  # noqa: E402
from plots import loss_curves, trial_outputs  # noqa: E402
from strides import step_times  # noqa: E402

TRAIN_SESSIONS = ("A", "B")
TEST_SESSIONS = ("C",)
PEAK_WEIGHT = 1.0            # peak penalty vs count loss (the count loss starts ~1000x bigger)
MAX_GRAD_NORM = 1.0          # clip each update: the count loss starts out huge (see docs)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split(trials, val_frac, seed, train_sessions=TRAIN_SESSIONS):
    """Whole trials only. Validation takes the same share of each kind of
    trial (session x speed, zero-step trials on their own) so it isn't, say,
    all jogging."""
    rng = random.Random(seed)
    groups = {}
    for tr in trials:
        if tr.session in train_sessions and tr.counted is not None:
            key = (tr.session, "zero" if tr.counted == 0 else tr.meta.get("speed"))
            groups.setdefault(key, []).append(tr)
    train, val = [], []
    for key in sorted(groups, key=str):
        g = sorted(groups[key], key=lambda tr: tr.name)
        rng.shuffle(g)
        n_val = max(1, round(val_frac * len(g))) if len(g) >= 2 else 0
        val += g[:n_val]
        train += g[n_val:]
    test = [tr for tr in trials if tr.session in TEST_SESSIONS]
    ignored = [tr.name for tr in trials if tr.session not in tuple(train_sessions) + TEST_SESSIONS
               or (tr.session in train_sessions and tr.counted is None)]
    return train, val, test, ignored


def norm_stats(train):
    allx = np.concatenate([tr.x for tr in train])
    mean, std = allx.mean(axis=0), allx.std(axis=0)
    return mean, np.where(std < 1e-6, 1.0, std)


def batches(trials, mean, std, device, batch_size, shuffle, rng):
    """Pad trials of different lengths into (B, 8, N) batches with a mask."""
    order = list(range(len(trials)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [trials[j] for j in order[i:i + batch_size]]
        n = max(len(tr.x) for tr in chunk)
        x = np.zeros((len(chunk), len(CHANNELS), n), dtype=np.float32)   # padding = 0 = the mean
        mask = np.zeros((len(chunk), n), dtype=np.float32)
        for k, tr in enumerate(chunk):
            x[k, :, :len(tr.x)] = ((tr.x - mean) / std).T
            mask[k, :len(tr.x)] = 1
        yield chunk, torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)


def predict(model, trials, mean, std, device):
    model.eval()
    out = []
    with torch.no_grad():
        for tr in trials:
            x = torch.from_numpy(((tr.x - mean) / std).T[None].astype(np.float32)).to(device)
            out.append(model(x)[0].cpu().numpy())
    return out


def val_errors(model, val, mean, std, device):
    ys = predict(model, val, mean, std, device)
    peak = np.mean([abs(len(peak_steps(y)) - tr.counted) for y, tr in zip(ys, val)])
    total = np.mean([abs(sum_count(y) - tr.counted) for y, tr in zip(ys, val)])
    return float(peak), float(total)


def train(model, train_set, val, mean, std, device, args, log):
    rng = random.Random(args.seed)
    history = {"pretrain": [], "finetune": {"train_count": [], "val_peak_mae": [], "val_sum_mae": []}}
    pseudo = {tr.name: step_times(tr.gy) for tr in train_set}

    # ---- warm start: learn "a step looks like this" from the gyro detector ----
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.pretrain_epochs):
        model.train()
        losses = []
        for chunk, x, mask in batches(train_set, mean, std, device, args.batch, True, rng):
            tw = [pseudo_targets(x.shape[2], pseudo[tr.name], device) for tr in chunk]
            target, weight = torch.stack([a for a, _ in tw]), torch.stack([b for _, b in tw])
            loss = pseudo_label_loss(model(x), mask, target, weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
            losses.append(loss.item())
        history["pretrain"].append(float(np.mean(losses)))
        log(f"  warm start {epoch + 1}/{args.pretrain_epochs}: pseudo-label loss {history['pretrain'][-1]:.3f}")

    # ---- fine-tune on the real labels: how many steps each trial had ----
    # A 10x smaller step than the warm start: Adam's first updates move every
    # weight by about lr at once, and at 1e-3 that alone wipes out the peaks.
    opt = torch.optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=1e-4)
    best = (float("inf"), float("inf"))
    best_state, best_epoch, waited = None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for chunk, x, mask in batches(train_set, mean, std, device, args.batch, True, rng):
            counted = torch.tensor([float(tr.counted) for tr in chunk], device=device)
            y = model(x)
            c = count_loss(y, mask, counted)
            loss = c + PEAK_WEIGHT * peak_penalty(y, mask)
            opt.zero_grad()
            loss.backward()          # backprop: gradient of the loss w.r.t. every weight
            # The first fine-tune batches are off by ~100+ steps, so (sum - counted)^2
            # gives enormous gradients; one full-size step would drive every output to 0
            # and the sigmoid would never recover. Clipping keeps each update small.
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()               # Adam nudges each weight against its gradient
            losses.append(c.item())
        peak_mae, sum_mae = val_errors(model, val, mean, std, device)
        ft = history["finetune"]
        ft["train_count"].append(float(np.mean(losses)))
        ft["val_peak_mae"].append(peak_mae)
        ft["val_sum_mae"].append(sum_mae)
        if (peak_mae, sum_mae) < best:
            best, best_epoch, waited = (peak_mae, sum_mae), epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        if epoch % args.log_every == 0 or epoch == 1:
            log(f"  fine-tune {epoch}: train count loss {ft['train_count'][-1]:.2f}, "
                f"val error {peak_mae:.2f} steps (peaks), {sum_mae:.2f} (sum)")
        if waited >= args.patience:
            log(f"  stopped at epoch {epoch}: no better validation count for {args.patience} epochs")
            break
    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    log(f"  kept epoch {best_epoch}: val error {best[0]:.2f} steps (peaks), {best[1]:.2f} (sum)")
    return history


def evaluate(model, splits, mean, std, device):
    rows, outputs = [], {}
    for split_name, trials in splits.items():
        for tr, y in zip(trials, predict(model, trials, mean, std, device)):
            peaks, base = peak_steps(y), step_times(tr.gy)
            true = true_step_samples(tr)
            row = {"trial": tr.name, "split": split_name, "speed": tr.meta.get("speed"),
                   "type": tr.meta.get("type"), "counted": tr.counted,
                   "true": len(true) if true is not None else None,
                   "gyro": len(base), "cnn_peaks": len(peaks), "cnn_sum": round(sum_count(y), 2)}
            if true is not None:
                row["f1_cnn"] = round(event_f1(peaks, true)[2], 3)
                row["f1_gyro"] = round(event_f1(base, true)[2], 3)
            rows.append(row)
            outputs[tr.name] = dict(trial=tr, y=y, peaks=peaks, baseline=base, true=true)
    return pd.DataFrame(rows), outputs


def summary(df):
    """Mean absolute count error per split (and per speed on the test set)."""
    def block(d):
        out = {"trials": len(d), "steps (counted)": int(d["counted"].sum())}
        for col, label in (("gyro", "gyro detector"), ("cnn_peaks", "CNN peaks"), ("cnn_sum", "CNN sum")):
            err = (d[col] - d["counted"]).abs()
            out[f"{label} MAE"] = round(err.mean(), 2)
            out[f"{label} total %"] = round(100 * (d[col].sum() - d["counted"].sum()) / max(d["counted"].sum(), 1), 1)
        if d["true"].notna().all():
            out["CNN peaks MAE vs true"] = round((d["cnn_peaks"] - d["true"]).abs().mean(), 2)
            out["F1 CNN"] = round(d["f1_cnn"].mean(), 3)
            out["F1 gyro"] = round(d["f1_gyro"].mean(), 3)
        return out
    rows = {name: block(d) for name, d in df.groupby("split", sort=False)}
    test = df[df["split"] == "test"]
    for speed, d in test.groupby("speed"):
        rows[f"test: {speed}"] = block(d)
    return pd.DataFrame(rows).T


def markdown(df):
    """A DataFrame as a Markdown table (no extra dependency)."""
    cells = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells])


def main():
    ap = argparse.ArgumentParser(description="Train the CNN step detector.")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "sim" / "raw")
    ap.add_argument("--truth-dir", type=Path,
                    help="true step times (simulator only); default: ../truth next to --data-dir if it exists")
    ap.add_argument("--tag", help="name for the outputs (default: sim if --data-dir is under data/sim, else real)")
    ap.add_argument("--train-sessions", default=",".join(TRAIN_SESSIONS),
                    help="sessions for train/validation, comma-separated (case matters); C is always test only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    # Training length is set in optimizer updates (batches), not epochs, so a
    # small real dataset (2 batches per epoch) trains as long as the bigger
    # simulated one (6 per epoch). Epochs = updates / batches per epoch.
    ap.add_argument("--pretrain-updates", type=int, default=90, help="warm-start length")
    ap.add_argument("--max-updates", type=int, default=2400, help="fine-tune length at most")
    ap.add_argument("--patience-updates", type=int, default=600,
                    help="stop fine-tuning after this long without a better validation count")
    ap.add_argument("--lr", type=float, default=3e-3, help="warm-start learning rate")
    ap.add_argument("--ft-lr", type=float, default=3e-4, help="fine-tune learning rate")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10, help="print progress every this many epochs")
    ap.add_argument("--device", default="auto", help="auto (mps, then cuda, then cpu), or a torch device")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    is_sim = "sim" in data_dir.relative_to(REPO).parts if data_dir.is_relative_to(REPO) else False
    tag = args.tag or ("sim" if is_sim else "real")
    truth_dir = args.truth_dir or (data_dir.parent / "truth" if (data_dir.parent / "truth").is_dir() else None)
    if truth_dir and not is_sim:
        truth_dir = None                    # real data has no true step times
    out = REPO / "models" / "step_detector" / tag
    figs = REPO / "docs" / "figures" / "step_detector" / tag
    log_lines = []

    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(msg)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    trials, skipped = load_trials(data_dir, truth_dir)
    train_set, val, test, ignored = split(trials, args.val_frac, args.seed, args.train_sessions.split(","))
    log(f"data: {data_dir.relative_to(REPO) if data_dir.is_relative_to(REPO) else data_dir}"
        f"{'  (SIMULATED)' if is_sim else ''}; {len(trials)} trials, skipped {len(skipped)} bench_*"
        f"{', ignored ' + ', '.join(ignored) if ignored else ''}")
    log(f"split by trial: train {len(train_set)}, val {len(val)}, test (session C) {len(test)}")
    if not train_set or not val:
        raise SystemExit("need session A/B trials with counted_steps for training and validation")
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    despiked = sum(tr.despiked for tr in trials)
    if despiked:
        log(f"despiked {despiked} glitch samples")

    per_epoch = math.ceil(len(train_set) / args.batch)
    args.pretrain_epochs = math.ceil(args.pretrain_updates / per_epoch)
    args.epochs = math.ceil(args.max_updates / per_epoch)
    args.patience = math.ceil(args.patience_updates / per_epoch)
    log(f"{per_epoch} batches per epoch: warm start {args.pretrain_epochs} epochs, fine-tune up to "
        f"{args.epochs}, patience {args.patience}")

    mean, std = norm_stats(train_set)
    model = StepCNN(in_channels=len(CHANNELS)).to(device)
    log(f"model: {n_parameters(model)} parameters, receptive field {receptive_field()} samples "
        f"({receptive_field() / 50:.1f} s), device {device}")

    history = train(model, train_set, val, mean, std, device, args, log)

    splits = {"train": train_set, "val": val, "test": test}
    df, outputs = evaluate(model, splits, mean, std, device)
    table = summary(df)
    log("\n" + table.to_string())

    # ---- save ----
    torch.save({"state_dict": model.state_dict(), "channels": CHANNELS,
                "config": {"width": model.head.in_channels}}, out / "model.pt")
    (out / "norm.json").write_text(json.dumps({
        "channels": CHANNELS, "mean": np.round(mean, 6).tolist(), "std": np.round(std, 6).tolist(),
        "us_clip_mm": US_CLIP_MM, "us_lag": "ultrasonic moved back us_lag_samples rows before scaling",
        "from": f"{len(train_set)} training trials"}, indent=1) + "\n")
    (out / "history.json").write_text(json.dumps(history) + "\n")
    df.to_csv(out / "results.csv", index=False)
    source = "SIMULATED data (scripts/make_sim_dataset.py): pipeline test only" if is_sim else "real trials"
    (out / "results.md").write_text(
        f"# Step detector results ({tag})\n\nData: {source}, {data_dir.name}/. "
        f"Train {len(train_set)}, val {len(val)}, test {len(test)} trials. "
        f"Command: `{' '.join(['python training/step_detector/train.py'] + sys.argv[1:])}`. "
        f"Kept epoch {history['best_epoch']} of up to {args.epochs}.\n\nMAE = mean absolute count error per trial vs counted_steps; "
        f"total % = summed count vs summed counted steps.\n\n"
        + markdown(table.reset_index(names="")) + "\n\n## Per trial\n\n" + markdown(df) + "\n")

    label = "SIMULATED data" if is_sim else "real data"
    loss_curves(history, figs / "loss_curves.png", f"Step detector training ({label})")
    # a few held-out trials: one per speed plus the back-and-forth from the test
    # set, or from validation while there is no session C yet
    pool, which = (test, "test") if test else (val, "validation")
    show = []
    for speed in ("slow", "normal", "fast", "jog"):
        show += [tr for tr in pool if tr.meta.get("speed") == speed and tr.counted][:1]
    show += [tr for tr in pool if "back-and-forth" in (tr.meta.get("notes") or "")][:1]
    show += [tr for tr in val if tr.counted == 0][:1]
    if show:
        trial_outputs([outputs[tr.name] for tr in show], figs / f"{which}_trials.png",
                      f"{label}; {which} trials (last block: a zero-step validation trial)")
    log(f"\nsaved {out.relative_to(REPO)}/ (model.pt, norm.json, results.md/.csv, history.json) "
        f"and {figs.relative_to(REPO)}/")
    (out / "train_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
