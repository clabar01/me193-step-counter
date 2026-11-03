"""Export the trained models for the board: plain numpy arrays, no PyTorch.

Reads   models/step_detector/<tag>/model.pt + norm.json
        models/step_length/<tag>/model.pt + norm.json
Writes  models/step_detector.npz, models/step_length.npz
and copies both into board/python/models/, which board/deploy_board.sh ships
(it refuses to deploy if a copy differs from models/).

board/python/live_steps.py runs them with numpy; test_live_steps.py checks it
gives the same outputs as the PyTorch models.

Usage:
    python scripts/export_models.py              # the real-data models
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
APP_MODELS = REPO / "board" / "python" / "models"


def main():
    ap = argparse.ArgumentParser(description="Export the step detector and step-length models as numpy .npz.")
    ap.add_argument("--tag", default="real")
    args = ap.parse_args()

    det_dir = REPO / "models" / "step_detector" / args.tag
    sd = torch.load(det_dir / "model.pt", map_location="cpu")["state_dict"]
    norm = json.loads((det_dir / "norm.json").read_text())
    convs = sorted({k.split(".")[1] for k in sd if k.startswith("body.") and k.endswith(".weight")}, key=int)
    det = {"channels": np.array(norm["channels"]), "mean": np.array(norm["mean"], np.float32),
           "std": np.array(norm["std"], np.float32), "us_clip_mm": np.array(norm["us_clip_mm"]),
           "dilations": np.array([1, 2, 4, 8]), "head_w": sd["head.weight"].numpy()[:, :, 0],
           "head_b": sd["head.bias"].numpy()}
    for i, k in enumerate(convs):
        det[f"conv{i}_w"] = sd[f"body.{k}.weight"].numpy()
        det[f"conv{i}_b"] = sd[f"body.{k}.bias"].numpy()

    len_dir = REPO / "models" / "step_length" / args.tag
    sl = torch.load(len_dir / "model.pt", map_location="cpu")
    lnorm = json.loads((len_dir / "norm.json").read_text())
    lin = sorted({k.split(".")[1] for k in sl["state_dict"] if k.endswith(".weight")}, key=int)
    length = {"features": np.array(lnorm["features"]), "mean": np.array(lnorm["mean"], np.float32),
              "std": np.array(lnorm["std"], np.float32)}
    for i, k in enumerate(lin):
        length[f"lin{i}_w"] = sl["state_dict"][f"net.{k}.weight"].numpy()
        length[f"lin{i}_b"] = sl["state_dict"][f"net.{k}.bias"].numpy()

    APP_MODELS.mkdir(parents=True, exist_ok=True)
    for name, arrays in (("step_detector.npz", det), ("step_length.npz", length)):
        out = REPO / "models" / name
        with open(out, "wb") as f:           # fixed bytes for the same weights (no timestamps)
            np.savez(f, **arrays)
        shutil.copy2(out, APP_MODELS / name)
        print(f"{out.relative_to(REPO)} ({out.stat().st_size // 1024} KB) -> {(APP_MODELS / name).relative_to(REPO)}")


if __name__ == "__main__":
    main()
