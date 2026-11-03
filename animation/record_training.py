"""Record the real training runs for the explainer animation.

Re-runs the real-data step-detector training exactly as
`python training/step_detector/train.py --data-dir data/raw` does (same seed,
settings and order of random draws) and the step-length MLP's leave-one-out
fold for B_03, and saves what the animation shows into animation/data/story.json:

  - one validation trial's 8 input channels (A_09, 30 counted steps)
  - the step detector's output on it at points in training (warm start, fine-tune)
  - the gyro pseudo-labels, the loss history, a few weights over training,
    per-layer gradient sizes at a few epochs
  - the final model's feature maps (each layer) on a 4 s window
  - B_03's per-step features, and the MLP's step lengths / loss / gradients
    at points in its training (the fold where B_03 is held out)

Then it checks the re-run's final step detector equals the committed
models/step_detector/real/model.pt, so the animation shows the real run.
Nothing in models/ is written.

Usage: python animation/record_training.py
"""

import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training" / "step_detector"))
import train as det  # noqa: E402
from counting import peak_steps, step_mass  # noqa: E402
from data import CHANNELS, load_trials  # noqa: E402
from losses import count_loss, peak_penalty  # noqa: E402
from model import StepCNN  # noqa: E402
from strides import step_times  # noqa: E402

OUT = REPO / "animation" / "data" / "story.json"
SHOW_TRIAL = "A_09"
WINDOW = (500, 700)                   # samples of SHOW_TRIAL for the feature maps (10-14 s)
WARM_SNAPS = {0, 1, 3, 10, 45}
FINE_SNAPS = {1, 5, 20, 50, 100, 200, 400}
GRAD_SNAPS = {1, 50, 400}
TRACKED = [("body.0.weight", (0, 4, 4)), ("body.0.weight", (5, 4, 4)), ("body.0.weight", (9, 0, 4)),
           ("body.6.weight", (3, 7, 4)), ("head.weight", (0, 2, 0)), ("head.weight", (0, 11, 0))]
LENGTH_TRIAL = "B_03"
LENGTH_SNAPS = {0, 10, 50, 200, 600, 1499}


def r(a, n=3):
    return np.round(np.asarray(a, dtype=float), n).tolist()


def record_detector():
    # exactly train.py main()'s defaults and order, for --data-dir data/raw
    args = Namespace(lr=3e-3, ft_lr=3e-4, batch=8, seed=0, log_every=10 ** 9)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = det.pick_device("auto")
    trials, _ = load_trials(REPO / "data" / "raw")
    train_set, val, _, _ = det.split(trials, 0.2, args.seed)
    per_epoch = -(-len(train_set) // args.batch)
    args.pretrain_epochs, args.epochs, args.patience = -(-90 // per_epoch), -(-2400 // per_epoch), -(-600 // per_epoch)
    mean, std = det.norm_stats(train_set)
    model = StepCNN(in_channels=len(CHANNELS)).to(device)
    show = next(tr for tr in trials if tr.name == SHOW_TRIAL)
    assert show in val, f"{SHOW_TRIAL} should be a validation trial"

    snaps, weights, grads = [], {"warm": [], "fine": []}, []

    def layer_grads(m):
        """Per-layer gradient size of the fine-tune loss over all training trials (no update)."""
        m.zero_grad()
        for chunk, x, mask in det.batches(train_set, mean, std, device, args.batch, False, None):
            y = m(x)
            counted = torch.tensor([float(t.counted) for t in chunk], device=device)
            (count_loss(y, mask, counted) + det.PEAK_WEIGHT * peak_penalty(y, mask)).backward()
        out = {name: float(p.grad.norm()) for name, p in m.named_parameters() if name.endswith("weight")}
        m.zero_grad()
        return out

    def on_epoch(stage, epoch, m):
        sd = m.state_dict()
        weights[stage].append([float(sd[name][idx]) for name, idx in TRACKED])
        if (stage == "warm" and epoch in WARM_SNAPS) or (stage == "fine" and epoch in FINE_SNAPS):
            y = det.predict(m, [show], mean, std, device)[0]
            snaps.append({"stage": stage, "epoch": epoch, "y": r(y), "count": int(len(peak_steps(y))),
                          "sum": round(float(y.sum()), 2)})
        if stage == "fine" and epoch in GRAD_SNAPS:
            grads.append({"epoch": epoch, "norms": layer_grads(m)})
        m.train()

    history = det.train(model, train_set, val, mean, std, device, args, lambda *_: None, on_epoch)
    y = det.predict(model, [show], mean, std, device)[0]
    snaps.append({"stage": "final", "epoch": history["best_epoch"], "y": r(y),
                  "count": int(len(peak_steps(y))), "sum": round(float(y.sum()), 2)})

    committed = torch.load(REPO / "models" / "step_detector" / "real" / "model.pt", map_location="cpu")["state_dict"]
    diff = max(float((model.state_dict()[k].cpu() - v).abs().max()) for k, v in committed.items())

    # feature maps of the final model on a window of the shown trial
    a, b = WINDOW
    z = torch.tensor(((show.x - mean) / std).T[None], dtype=torch.float32, device=device)
    maps, h = [], z
    with torch.no_grad():
        for layer in model.body:
            h = layer(h)
            if isinstance(layer, torch.nn.ReLU):
                maps.append(r(h[0, :, a:b].cpu().numpy(), 2))
    return {
        "trial": show.name, "counted": show.counted, "channels": CHANNELS,
        "x": r(show.x.T, 3), "pseudo_steps": step_times(show.gy).tolist(),
        "snapshots": snaps, "history": history, "tracked_weights": [f"{n}{list(i)}" for n, i in TRACKED],
        "weights": weights, "grads": grads, "window": [a, b], "feature_maps": maps,
        "final_steps": peak_steps(y).tolist(), "final_mass": r(step_mass(y)),
        "max_diff_vs_committed_model": diff, "train_trials": len(train_set), "val_trials": len(val),
    }


def record_length():
    # training/step_length has its own train.py and plots.py: load them under other names
    import importlib.util
    folder = REPO / "training" / "step_length"
    clashing = {k: sys.modules.pop(k) for k in ("train", "plots", "features") if k in sys.modules}
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("step_length_train", folder / "train.py")
    sl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sl)
    from features import FEATURES, distance_trials
    sys.path.remove(str(folder))
    sys.modules.update(clashing)
    data, _ = distance_trials(REPO / "data" / "raw")
    data = [d for d in data if len(d[1])]
    held = next(d for d in data if d[0].name == LENGTH_TRIAL)
    rest = [d for d in data if d is not held]
    feats, dists = [d[1] for d in rest], [d[0].meta["distance_m"] for d in rest]
    snaps, losses = [], []

    def on_update(step, model, loss, mean, std):
        losses.append(loss)
        if step in LENGTH_SNAPS:
            lengths = sl.predict_mlp(model, mean, std, held[1])
            grad = {n: float(p.grad.norm()) for n, p in model.named_parameters() if n.endswith("weight")}
            snaps.append({"step": step, "lengths": r(lengths), "total": round(float(lengths.sum()), 2),
                          "loss": loss, "grads": grad})

    model, mean, std = sl.fit_mlp(feats, dists, 0, on_update)
    final = sl.predict_mlp(model, mean, std, held[1])
    ks = {m: sl.fit_scale([sl.baseline_terms(m, f, 1.0) for f in feats], dists) for m in ("fixed", "weinberg")}
    return {
        "trial": held[0].name, "distance_m": held[0].meta["distance_m"], "speed": held[0].meta["speed"],
        "features": FEATURES, "step_features": r(held[1]), "snapshots": snaps, "losses": r(losses, 6),
        "final_lengths": r(final), "final_total": round(float(final.sum()), 2),
        "fixed_total": round(ks["fixed"] * len(held[1]), 2),
        "weinberg_total": round(float(ks["weinberg"] * sl.baseline_terms("weinberg", held[1], 1.0).sum()), 2),
        "other_trials": [{"name": d[0].name, "distance_m": d[0].meta["distance_m"], "steps": len(d[1])} for d in rest],
        # leave-one-trial-out mean |distance error| % over all distance trials (models/step_length/real)
        "loo_mean_abs_err_pct": loo_summary(),
    }


def loo_summary():
    import pandas as pd
    df = pd.read_csv(REPO / "models" / "step_length" / "real" / "results.csv")
    return {m: round(float(df[f"{m}_err%"].abs().mean()), 1) for m in ("fixed", "weinberg", "pendulum", "mlp")}


def main():
    det_story = record_detector()
    print(f"step detector: re-run vs committed model, max weight difference {det_story['max_diff_vs_committed_model']:.2e}; "
          f"{SHOW_TRIAL} final count {det_story['snapshots'][-1]['count']} (counted {det_story['counted']})")
    len_story = record_length()
    print(f"step length: {LENGTH_TRIAL} predicted {len_story['final_total']} m vs {len_story['distance_m']} m measured")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"detector": det_story, "length": len_story}) + "\n")
    print(f"saved {OUT.relative_to(REPO)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
