"""The step detector: a small, fully convolutional 1D CNN.

Input: one whole trial, (batch, 8 channels, N samples), z-scored.
Output: (batch, N), one number in 0..1 per sample: "a step happens here".
Ideally it is ~1 at the one sample of each heel strike and ~0 elsewhere, so
counting steps is either counting the peaks or just adding everything up.

Fully convolutional (no pooling, no dense layer) means the output has one
value per input sample, any trial length works, and on the board it could run
on a rolling buffer the same way.
"""

import torch
from torch import nn

KERNEL = 9
DILATIONS = (1, 2, 4, 8)
WIDTH = 24


class StepCNN(nn.Module):
    def __init__(self, in_channels=8, width=WIDTH, kernel=KERNEL, dilations=DILATIONS):
        super().__init__()
        layers = []
        ch = in_channels
        for d in dilations:
            # Each Conv1d slides `width` learned filters, each `kernel` taps long,
            # across time. Dilation d spaces the taps d samples apart, so the
            # four layers look 0.16 s, then 0.32 s, 0.64 s and 1.28 s wide and
            # together see ~2.4 s, more than one stride, at little cost.
            # "Same" padding keeps the length: one output per input sample.
            layers += [nn.Conv1d(ch, width, kernel, dilation=d, padding=d * (kernel - 1) // 2), nn.ReLU()]
            ch = width
        self.body = nn.Sequential(*layers)
        # 1x1 conv: at each sample, a weighted sum of the 24 features -> one score
        self.head = nn.Conv1d(width, 1, 1)

    def forward(self, x):
        return torch.sigmoid(self.head(self.body(x))).squeeze(1)


def receptive_field(kernel=KERNEL, dilations=DILATIONS):
    """How many input samples one output sample can see."""
    return 1 + sum(d * (kernel - 1) for d in dilations)


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())
