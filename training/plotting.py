"""Shared figure style for the Phase 4/5 scripts.

Class colours are fixed per class (same as the dashboard), so a class looks the
same in every figure no matter which classes are present.
"""

import matplotlib.pyplot as plt
import numpy as np

CLASS_COLORS = {
    "standing": "#2a78d6",
    "walking": "#eb6834",
    "running": "#1baf7a",
    "biking": "#eda100",
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def style(ax, grid_axis="y"):
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def confusion(y_true, y_pred, classes):
    """Counts, rows = true class, columns = predicted class."""
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def plot_confusion(ax, cm, classes, title):
    """Row-normalised colour (one blue ramp), each cell labelled count + row %."""
    rows = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if rows[i, 0] == 0:
                continue
            ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if frac[i, j] > 0.6 else TEXT)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted", color=MUTED)
    ax.set_ylabel("true", color=MUTED)
    ax.set_title(title, color=TEXT, loc="left", fontsize=12)
    ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")
