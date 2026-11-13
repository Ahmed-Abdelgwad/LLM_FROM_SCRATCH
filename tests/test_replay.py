"""Pretraining replay inside fine-tuning (finetune/replay.py, D-042)."""

from pathlib import Path

import numpy as np
import pytest

from conftest import write_source
from scratchllm.finetune.chat import IGNORE, EncodedExample
from scratchllm.finetune.replay import ReplaySampler
from scratchllm.training.batches import load_source


@pytest.fixture()
def sources(tmp_path: Path):
    write_source(tmp_path, "alpha", base=0, train_tokens=3000)
    write_source(tmp_path, "beta", base=100, train_tokens=1000)
    return [load_source(tmp_path, "alpha"), load_source(tmp_path, "beta")]


@pytest.mark.parametrize(("fraction", "expected"), [(0.05, 1600), (0.1, 3200), (0.2, 6400)])
def test_exactly_the_registered_share_of_rows_is_replayed_spread_over_every_step(
    sources, fraction, expected
):
    s = ReplaySampler(sources, [3, 1], fraction, micro_batch_size=8, grad_accum_steps=4, seed=1)
    per_step = [sum(len(s.rows(step, m)) for m in range(4)) for step in range(1000)]
    assert sum(per_step) == expected  # 1,000 steps of 32 rows
    assert max(per_step) - min(per_step) <= 1  # evenly spread, never bunched
    assert all(r < 8 for step in range(5) for m in range(4) for r in s.rows(step, m))


def test_a_window_is_a_slice_of_a_train_file_and_its_next_token_targets(sources):
    s = ReplaySampler(sources, [3, 1], 0.5, micro_batch_size=4, grad_accum_steps=1, seed=7)
    name, ex = s.window(step=3, micro=0, row=1, length=40)
    assert len(ex.inputs) == len(ex.targets) == 40 and IGNORE not in ex.targets
    assert ex.targets[:-1] == ex.inputs[1:]  # every token is a target: language modelling
    stream = np.fromfile(next(x for x in sources if x.name == name).train_path, dtype="<u2")
    text = stream.tolist()
    start = next(i for i in range(len(text)) if text[i : i + 41] == ex.inputs + ex.targets[-1:])
    assert start >= 0
    assert s.window(3, 0, 1, 40) == (name, ex)  # a pure function of (seed, step, micro, row)
    assert s.window(3, 0, 2, 40) != (name, ex)


def test_the_source_mix_follows_the_weights(sources):
    s = ReplaySampler(sources, [3, 1], 0.5, micro_batch_size=4, grad_accum_steps=1, seed=0)
    names = [s.window(step, 0, 0, 10)[0] for step in range(2000)]
    assert abs(names.count("alpha") / 2000 - 0.75) < 0.03


def test_mixing_keeps_the_shape_and_the_other_rows(sources):
    s = ReplaySampler(sources, [1, 1], 0.25, micro_batch_size=8, grad_accum_steps=2, seed=0)
    chat = [EncodedExample([1] * (5 + i), [IGNORE] * 3 + [2] * (2 + i)) for i in range(8)]
    out, rows = s.mix(0, 1, chat)
    assert rows == s.rows(0, 1) and len(rows) == 2
    assert [len(e.inputs) for e in out] == [len(e.inputs) for e in chat]  # same tokens read
    assert all(out[i] is chat[i] for i in range(8) if i not in rows)


def test_bad_settings_are_refused(sources):
    for fraction in (0.0, 1.0):
        with pytest.raises(ValueError, match="fraction"):
            ReplaySampler(sources, [1, 1], fraction, 8, 4, 0)
    with pytest.raises(ValueError, match="weight"):
        ReplaySampler(sources, [1], 0.1, 8, 4, 0)
