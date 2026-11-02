"""Sliding windows over a token file: the training examples of a language model.

The model is trained to predict each token from the ones before it. From a stream of ids
t0 t1 t2 ... a window of `context_length` ids is the input, and the same window moved one place
to the right is the target, so the target at position i is the token that follows the input at
position i (Raschka, Build a Large Language Model From Scratch, chapter 2):

    input   = ids[start : start + T]
    target  = ids[start + 1 : start + T + 1]

`stride` is the distance between the starts of two windows. With stride = T the windows do not
overlap and every token is used once per pass; with a smaller stride the windows overlap and
tokens are seen several times, which is more data-hungry on a small corpus and never more
informative about unseen text.

Several files can be given (one per source and split). A window never crosses from one file to
the next, because the end of a file is not the end of a sentence. The files are read through
`numpy.memmap`, so only the windows used are loaded from disk.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from itertools import accumulate
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scratchllm.data.tokens import DTYPE


def window_count(n_tokens: int, context_length: int, stride: int) -> int:
    """How many full (input, target) windows fit: the last one needs T + 1 ids."""
    if n_tokens < context_length + 1:
        return 0
    return (n_tokens - context_length - 1) // stride + 1


class TokenWindows(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: Sequence[Path], context_length: int, stride: int | None = None):
        if context_length < 1:
            raise ValueError("context_length must be at least 1")
        self.context_length = context_length
        self.stride = context_length if stride is None else stride
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        self._arrays = [
            np.memmap(p, dtype=DTYPE, mode="r") if p.stat().st_size else np.empty(0, DTYPE)
            for p in paths
        ]
        counts = [window_count(len(a), context_length, self.stride) for a in self._arrays]
        self._ends = list(accumulate(counts))  # window index at which each file ends

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        which = bisect_right(self._ends, index)
        local = index - (self._ends[which - 1] if which else 0)
        start = local * self.stride
        chunk = self._arrays[which][start : start + self.context_length + 1].astype(np.int64)
        tensor = torch.from_numpy(chunk)
        return tensor[:-1], tensor[1:]


def make_loader(
    dataset: TokenWindows, batch_size: int, shuffle: bool, seed: int, drop_last: bool = True
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """A loader whose order depends only on `seed`, so a run can be repeated exactly."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, drop_last=drop_last
    )
