"""Learning-rate schedule: linear warmup, then cosine decay to a floor.

Why warmup: at the start the weights are random and Adam's statistics are not yet reliable, so a
full-size step can throw the model into a bad region it never leaves. Why decay: late in training
a smaller step lets the weights settle into the bottom of the valley instead of bouncing across it.
"""

from __future__ import annotations

import math


def lr_at(step: int, lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Learning rate for the optimizer step with 0-based index `step`."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps  # reaches exactly `lr` on the last warmup step
    if step >= max_steps:
        return min_lr
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)
