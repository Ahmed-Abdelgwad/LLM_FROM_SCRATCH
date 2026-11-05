"""Where training and validation batches come from, and how a model is scored on them.

Training batches are random windows of the token files, drawn as a pure function of
(seed, step, micro-batch index). Nothing about the data order is kept as state, so a run that is
stopped and resumed at step k sees exactly the batches an uninterrupted run would have seen from
step k on. Each sequence first picks a source with probability proportional to its weight, then a
window start uniformly inside that source's train tokens (every start is allowed: stride 1).

Validation is fixed and deterministic: for each source, a set of evenly spaced non-overlapping
windows of its val split. The same windows are used at every evaluation of every run, so val
losses can be compared across steps and across runs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scratchllm.data.tokens import DTYPE
from scratchllm.model.gpt import GPT
from scratchllm.training.dataset import TokenWindows


@dataclass(frozen=True)
class SourceTokens:
    """One source's token files, as described by its `<source>.tokens.json`."""

    name: str
    train_path: Path
    val_path: Path
    train_tokens: int
    train_chars: int
    val_tokens: int
    val_chars: int
    vocab_size: int
    tokenizer_sha256: str
    train_sha256: str
    val_sha256: str


def load_source(tokens_dir: Path, name: str) -> SourceTokens:
    meta_path = tokens_dir / f"{name}.tokens.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no token description for source {name!r} in {tokens_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("role") != "training" or not {"train", "val"} <= set(meta["splits"]):
        raise ValueError(f"source {name!r} has no train/val split (it is evaluation-only)")
    train, val = meta["splits"]["train"], meta["splits"]["val"]
    for info in (train, val):  # cheap integrity check: the file is as long as the description says
        size = (tokens_dir / info["file"]).stat().st_size
        if size != info["tokens"] * DTYPE.itemsize:
            raise ValueError(
                f"{info['file']}: {size} bytes, description says {info['tokens']} tokens"
            )
    return SourceTokens(
        name=name,
        train_path=tokens_dir / train["file"],
        val_path=tokens_dir / val["file"],
        train_tokens=train["tokens"],
        train_chars=train["chars"],
        val_tokens=val["tokens"],
        val_chars=val["chars"],
        vocab_size=meta["vocab_size"],
        tokenizer_sha256=meta["tokenizer_sha256"],
        train_sha256=train["sha256"],
        val_sha256=val["sha256"],
    )


def training_source_names(tokens_dir: Path) -> list[str]:
    names = []
    for meta_path in sorted(tokens_dir.glob("*.tokens.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("role") == "training":
            names.append(meta["source"])
    return names


class TrainSampler:
    def __init__(
        self,
        sources: Sequence[SourceTokens],
        weights: Sequence[float],
        context_length: int,
        micro_batch_size: int,
        seed: int,
    ) -> None:
        if len(sources) != len(weights) or not sources:
            raise ValueError("need one weight per source, and at least one source")
        total = float(sum(weights))
        self.probabilities = np.asarray([w / total for w in weights])
        self.datasets = [TokenWindows([s.train_path], context_length, stride=1) for s in sources]
        for s, d in zip(sources, self.datasets, strict=True):
            if len(d) == 0:
                raise ValueError(f"{s.name}: fewer than {context_length + 1} train tokens")
        self.micro_batch_size = micro_batch_size
        self.seed = seed

    def batch(self, step: int, micro: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(inputs, targets), each (micro_batch_size, context_length)."""
        rng = np.random.default_rng([self.seed, step, micro])
        which = rng.choice(len(self.datasets), size=self.micro_batch_size, p=self.probabilities)
        pairs = []
        for k in which:
            dataset = self.datasets[k]
            pairs.append(dataset[int(rng.integers(len(dataset)))])
        return torch.stack([x for x, _ in pairs]), torch.stack([y for _, y in pairs])


def validation_sets(
    sources: Sequence[SourceTokens], context_length: int, split: str = "val"
) -> dict[str, TokenWindows]:
    """For each source, its non-overlapping windows of `split` ("val", or "train" for a probe of
    the training data); `evaluate` picks evenly spaced ones."""
    out: dict[str, TokenWindows] = {}
    for s in sources:
        dataset = TokenWindows([s.val_path if split == "val" else s.train_path], context_length)
        if len(dataset) == 0:
            raise ValueError(f"{s.name}: fewer than {context_length + 1} {split} tokens")
        out[s.name] = dataset
    return out


def evenly_spaced(n_available: int, n_wanted: int) -> list[int]:
    """`n_wanted` indices spread over range(n_available), or all of them if fewer exist."""
    if n_wanted >= n_available:
        return list(range(n_available))
    return sorted({int(i) for i in np.linspace(0, n_available - 1, n_wanted)})


@torch.no_grad()
def evaluate(
    model: GPT,
    sets: dict[str, TokenWindows],
    batch_size: int,
    max_windows: int,
    device: torch.device,
    autocast: Callable[[], AbstractContextManager[Any]],
) -> dict[str, float]:
    """Mean cross-entropy (nats per token) of `model` on each source's fixed val windows."""
    was_training = model.training
    model.eval()
    try:
        return _mean_losses(model, sets, batch_size, max_windows, device, autocast)
    finally:
        model.train(was_training)


def _mean_losses(
    model: GPT,
    sets: dict[str, TokenWindows],
    batch_size: int,
    max_windows: int,
    device: torch.device,
    autocast: Callable[[], AbstractContextManager[Any]],
) -> dict[str, float]:
    losses: dict[str, float] = {}
    for name, dataset in sets.items():
        picks = evenly_spaced(len(dataset), max_windows)
        total = 0.0
        for start in range(0, len(picks), batch_size):
            chunk = [dataset[i] for i in picks[start : start + batch_size]]
            x = torch.stack([a for a, _ in chunk]).to(device)
            y = torch.stack([b for _, b in chunk]).to(device)
            with autocast():
                _, loss = model(x, y)
            assert loss is not None
            total += float(loss) * len(chunk)  # every window has the same number of tokens
        losses[name] = total / len(picks)
    return losses


def bits_per_char(loss_nats: float, tokens: int, chars: int) -> float:
    """Bits per character from a per-token loss, using the split's tokens-per-character ratio.

    The ratio is that of the whole val split, so it is exact only if the evaluated windows have
    the same density of tokens as the split; and the tokens include one end-of-text id per
    document, which the loss also has to predict.
    """
    return loss_nats / math.log(2) * tokens / chars
