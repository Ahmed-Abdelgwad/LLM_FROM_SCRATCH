"""Pretraining text replayed inside fine-tuning batches, at a fixed budget (D-042).

The remedy for forgetting named in D-039 and D-040: InstructGPT mixes pretraining gradients into
its fine-tuning ("PPO-ptx", Ouyang et al. 2022, eq. 2; appendix C.3 for SFT-style mixing), and
continual-learning work replays a small share of old data (Scialom et al. 2022, 1%; Ibrahim et
al. 2024, 1% to 25%). Here a share `fraction` of the rows of every step is pretraining text
instead of a chat example, and nothing else changes:

* the number of steps, the rows per step and the input tokens per step are those of the run
  without replay. A replay row takes the place of the chat example the pair sampler drew for that
  row and has exactly its length, so every micro-batch has the same shape and the same tokens
  read; the replaced example is simply not seen;
* which rows are replayed is fixed in advance: the rows are numbered in the order they are
  trained on, r = 0, 1, 2, ..., and row r is replayed when floor((r + 1) * fraction) exceeds
  floor(r * fraction). Over R rows exactly floor(R * fraction) are replayed, spread evenly, so
  5% of a run of 1,000 steps of 32 rows is exactly 1,600 rows;
* a replay row is a window of a pretraining source's TRAIN tokens (the validation tokens measure
  forgetting and are never trained on), the source drawn with the base model's own pretraining
  weights and the start uniformly, as a pure function of (seed, step, micro-batch, row), like
  every other batch of this project;
* every token of a replay row is a target (it is language-modelling text, scored as in
  pretraining). The step loss stays the mean over all scored tokens of the step, so the share of
  the loss that is replay is larger than `fraction`: chat rows score their response only. The
  trainer logs both shares.

With `fraction` = 0 no row is replayed and the trainer takes exactly its old path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from scratchllm.data.tokens import DTYPE
from scratchllm.finetune.chat import EncodedExample
from scratchllm.training.batches import SourceTokens

_REPLAY_STREAM = 0x5245504C  # "REPL": keeps these draws apart from the pair sampler's


class ReplaySampler:
    def __init__(
        self,
        sources: Sequence[SourceTokens],
        weights: Sequence[float],
        fraction: float,
        micro_batch_size: int,
        grad_accum_steps: int,
        seed: int,
    ) -> None:
        if not 0 < fraction < 1:
            raise ValueError("the replay fraction must be in (0, 1)")
        if len(sources) != len(weights) or not sources or any(w <= 0 for w in weights):
            raise ValueError("need one positive weight per source, and at least one source")
        self.fraction = Fraction(str(fraction))  # exact: 0.05 is 1/20, not a nearby float
        total = float(sum(weights))
        self.probabilities = np.asarray([w / total for w in weights])
        self.names = [s.name for s in sources]
        self.arrays = [np.memmap(s.train_path, dtype=DTYPE, mode="r") for s in sources]
        self.micro_batch_size, self.grad_accum_steps, self.seed = (
            micro_batch_size, grad_accum_steps, seed,
        )  # fmt: skip

    def replayed(self, row: int) -> bool:
        """Whether global row `row` (0-based, in training order) is replayed."""
        return math.floor((row + 1) * self.fraction) > math.floor(row * self.fraction)

    def rows(self, step: int, micro: int) -> list[int]:
        """The rows of this micro-batch (0 .. micro_batch_size - 1) that are replayed."""
        first = (step * self.grad_accum_steps + micro) * self.micro_batch_size
        return [i for i in range(self.micro_batch_size) if self.replayed(first + i)]

    def window(self, step: int, micro: int, row: int, length: int) -> tuple[str, EncodedExample]:
        """A pretraining window of `length` tokens: (source, example scoring every token)."""
        rng = np.random.default_rng([self.seed, step, micro, row, _REPLAY_STREAM])
        k = int(rng.choice(len(self.arrays), p=self.probabilities))
        tokens = self.arrays[k]
        if len(tokens) < length + 1:
            raise ValueError(f"{self.names[k]}: fewer than {length + 1} train tokens")
        start = int(rng.integers(len(tokens) - length))
        ids = tokens[start : start + length + 1].astype(np.int64).tolist()
        return self.names[k], EncodedExample(inputs=ids[:-1], targets=ids[1:])

    def mix(
        self, step: int, micro: int, examples: Sequence[EncodedExample]
    ) -> tuple[list[EncodedExample], list[int]]:
        """`examples` with the replayed rows swapped for pretraining windows of the same length,
        and the indices of the swapped rows."""
        out = list(examples)
        rows = self.rows(step, micro)
        for i in rows:
            _, out[i] = self.window(step, micro, i, len(examples[i].inputs))
        return out, rows
