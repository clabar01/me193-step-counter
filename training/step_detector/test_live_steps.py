"""Check the board's numpy live pipeline (board/python/live_steps.py) against the
offline PyTorch one, on the real trials in data/raw (skipped if there are none).

  1. numpy CNN == PyTorch CNN on a whole trial
  2. streaming a trial sample by sample, processing every 0.25 s like the
     board, finds the same steps as the offline peak count (except within
     ~1.3 s of the start and end, which live never finalizes)
  3. numpy step-length MLP == PyTorch MLP

Run: python training/step_detector/test_live_steps.py   (after scripts/export_models.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "board" / "python"))
from counting import peak_steps  # noqa: E402
from data import load_trials  # noqa: E402
from live_steps import FINAL_LAG, LiveSteps, StepLengthMLP  # noqa: E402
from model import StepCNN  # noqa: E402

RAW = REPO / "data" / "raw"
APP_MODELS = REPO / "board" / "python" / "models"


def torch_model():
    ckpt = torch.load(REPO / "models" / "step_detector" / "real" / "model.pt", map_location="cpu")
    m = StepCNN()
    m.load_state_dict(ckpt["state_dict"])
    return m.eval()


def test_numpy_cnn_matches_torch(trials, live):
    m = torch_model()
    tr = trials[0]
    z = ((tr.x - live.cnn.mean) / live.cnn.std).T.astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(z[None]))[0].numpy()
    assert np.abs(live.cnn(z) - ref).max() < 1e-4


def test_streaming_finds_the_offline_steps(trials, live):
    m = torch_model()
    total_off = total_live = 0
    for tr in trials:
        z = ((tr.x - live.cnn.mean) / live.cnn.std).T.astype(np.float32)
        with torch.no_grad():
            offline = peak_steps(m(torch.from_numpy(z[None]))[0].numpy())
        df = pd.read_csv(RAW / tr.meta["csv"])
        lv = LiveSteps(APP_MODELS)
        found = []
        for i, row in enumerate(df.to_dict("records")):
            row["t"] = row.pop("t_ms")
            lv.add(row)
            if i % 12 == 11 or i == len(df) - 1:          # ~every 0.25 s
                found += [s["t"] for s in lv.process()["steps"]]
        found = np.searchsorted(df["t_ms"].to_numpy(), found)
        inner = offline[(offline >= FINAL_LAG + 8) & (offline < len(df) - FINAL_LAG - 8)]
        assert len(found) == len(inner), (tr.name, len(found), len(inner))
        assert not len(found) or np.abs(found - inner).max() <= 1, tr.name
        total_off += len(offline)
        total_live += len(found)
    print(f"    {len(trials)} trials: live {total_live} steps = offline {total_off} minus edge steps")


def test_numpy_mlp_matches_torch():
    sys.path.insert(0, str(REPO / "training" / "step_length"))
    from train import StepLengthMLP as TorchMLP
    ckpt = torch.load(REPO / "models" / "step_length" / "real" / "model.pt", map_location="cpu")
    tm = TorchMLP()
    tm.load_state_dict(ckpt["state_dict"])
    nm = StepLengthMLP(APP_MODELS / "step_length.npz")
    rng = np.random.default_rng(0)
    for _ in range(20):
        f = nm.mean + nm.std * rng.normal(size=len(nm.mean))
        with torch.no_grad():
            ref = tm(torch.tensor((f - nm.mean) / nm.std, dtype=torch.float32)[None])[0].item()
        assert abs(nm(f) - ref) < 1e-5


if __name__ == "__main__":
    if not any(RAW.glob("*.json")):
        print("no real trials in data/raw: skipped")
        sys.exit(0)
    trials, _ = load_trials(RAW)
    live = LiveSteps(APP_MODELS)
    test_numpy_cnn_matches_torch(trials, live)
    print("ok  test_numpy_cnn_matches_torch")
    test_streaming_finds_the_offline_steps(trials, live)
    print("ok  test_streaming_finds_the_offline_steps")
    test_numpy_mlp_matches_torch()
    print("ok  test_numpy_mlp_matches_torch")
