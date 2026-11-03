import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import VOCAB, write_source
from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.training.batches import (
    TrainSampler,
    bits_per_char,
    evaluate,
    evenly_spaced,
    load_source,
    training_source_names,
    validation_sets,
)

T = 16


def sources(tokens_dir: Path):
    return [load_source(tokens_dir, "alpha"), load_source(tokens_dir, "beta")]


def sampler(tokens_dir: Path, weights=(1.0, 1.0), seed: int = 3, batch: int = 6) -> TrainSampler:
    return TrainSampler(sources(tokens_dir), weights, T, batch, seed)


# ---------- reading the descriptions ----------


def test_sources_are_read_from_their_descriptions(tokens_dir: Path):
    alpha = load_source(tokens_dir, "alpha")
    assert (alpha.train_tokens, alpha.val_tokens, alpha.vocab_size) == (4000, 700, VOCAB)
    assert training_source_names(tokens_dir) == ["alpha", "beta"]


def test_evaluation_only_and_missing_sources_are_refused(tokens_dir: Path):
    meta = json.loads((tokens_dir / "alpha.tokens.json").read_text())
    meta.update(role="evaluation-only", splits={"eval": meta["splits"]["val"]})
    (tokens_dir / "alpha.tokens.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="evaluation-only"):
        load_source(tokens_dir, "alpha")
    assert training_source_names(tokens_dir) == ["beta"]
    with pytest.raises(FileNotFoundError):
        load_source(tokens_dir, "gamma")


def test_a_truncated_token_file_is_detected(tokens_dir: Path):
    path = tokens_dir / "alpha.train.bin"
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="bytes"):
        load_source(tokens_dir, "alpha")


# ---------- training batches ----------


def test_a_batch_is_a_pure_function_of_seed_step_and_micro_batch(tokens_dir: Path):
    a, b = sampler(tokens_dir), sampler(tokens_dir)
    x1, y1 = a.batch(5, 1)
    x2, y2 = b.batch(5, 1)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert not torch.equal(x1, a.batch(6, 1)[0])  # another step
    assert not torch.equal(x1, a.batch(5, 0)[0])  # another micro-batch
    assert not torch.equal(x1, sampler(tokens_dir, seed=4).batch(5, 1)[0])  # another seed
    # asking for step 9 first must not change step 5: there is no hidden state
    c = sampler(tokens_dir)
    c.batch(9, 0)
    assert torch.equal(c.batch(5, 1)[0], x1)


def test_targets_are_inputs_shifted_by_one_and_shapes_are_right(tokens_dir: Path):
    x, y = sampler(tokens_dir).batch(0, 0)
    assert x.shape == y.shape == (6, T) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    # the stream is periodic, so the next id is the next one in the cycle
    assert torch.equal(y, torch.where((x % 8) == 7, x - 7, x + 1))


def test_sources_are_chosen_by_weight(tokens_dir: Path):
    only_alpha = sampler(tokens_dir, weights=(1.0, 1e-12))
    assert all(int(only_alpha.batch(s, 0)[0].max()) < 32 for s in range(30))
    only_beta = sampler(tokens_dir, weights=(1e-12, 1.0))
    assert all(int(only_beta.batch(s, 0)[0].min()) >= 32 for s in range(30))
    mixed = sampler(tokens_dir, weights=(3.0, 1.0), batch=100)
    share_alpha = float(
        np.mean([(mixed.batch(s, 0)[0][:, 0] < 32).float().mean() for s in range(20)])
    )
    assert 0.70 < share_alpha < 0.80  # 3 : 1


def test_windows_start_at_any_offset_not_only_at_multiples_of_the_context(tokens_dir: Path):
    starts = {int(x[0]) % 8 for s in range(10) for x in sampler(tokens_dir).batch(s, 0)[0]}
    assert len(starts) == 8


def test_a_source_too_short_for_one_window_is_refused(tokens_dir: Path):
    write_source(tokens_dir, "tiny", base=0, train_tokens=T, val_tokens=700)
    with pytest.raises(ValueError, match="fewer than"):
        TrainSampler([load_source(tokens_dir, "tiny")], [1.0], T, 2, 0)
    with pytest.raises(ValueError):
        TrainSampler(sources(tokens_dir), [1.0], T, 2, 0)  # one weight for two sources


# ---------- validation ----------


def tiny_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=VOCAB, context_length=T, d_model=32, n_layers=1, n_heads=4))


def test_evenly_spaced_indices():
    assert evenly_spaced(10, 20) == list(range(10))
    assert evenly_spaced(100, 5) == [0, 24, 49, 74, 99]
    assert evenly_spaced(3, 3) == [0, 1, 2]
    assert len(evenly_spaced(1000, 7)) == 7


def test_evaluation_is_deterministic_per_source_and_does_not_leave_eval_mode(tokens_dir: Path):
    sets = validation_sets(sources(tokens_dir), T)
    model = tiny_model().train()
    first = evaluate(model, sets, 4, 8, torch.device("cpu"), nullcontext)
    second = evaluate(model, sets, 4, 8, torch.device("cpu"), nullcontext)
    assert first == second and set(first) == {"alpha", "beta"}
    assert model.training  # the mode it was in comes back
    assert abs(first["alpha"] - math.log(VOCAB)) < 0.5  # untrained: about a uniform guess
    model.eval()
    evaluate(model, sets, 4, 8, torch.device("cpu"), nullcontext)
    assert not model.training


def test_evaluation_matches_a_loss_computed_by_hand(tokens_dir: Path):
    sets = validation_sets(sources(tokens_dir), T)
    model = tiny_model().eval()
    picks = evenly_spaced(len(sets["alpha"]), 5)
    manual = []
    with torch.no_grad():
        for i in picks:
            x, y = sets["alpha"][i]
            manual.append(model(x[None], y[None])[1].item())
    got = evaluate(model, sets, 2, 5, torch.device("cpu"), nullcontext)  # batches of 2, 2, 1
    assert got["alpha"] == pytest.approx(sum(manual) / 5, rel=1e-5)


def test_evaluation_windows_do_not_overlap(tokens_dir: Path):
    sets = validation_sets(sources(tokens_dir), T)
    assert len(sets["alpha"]) == (700 - 1) // T  # stride = context length


def test_bits_per_character():
    assert bits_per_char(math.log(2), tokens=10, chars=40) == pytest.approx(0.25)
    assert bits_per_char(math.log(2) * 3, tokens=1, chars=1) == pytest.approx(3.0)
