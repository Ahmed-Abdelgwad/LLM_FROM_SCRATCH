"""The key/value cache (D-030): numerically equivalent to recomputing the window, not bitwise.

What is tested: the logits agree to 1e-5 in float32, and in float32 on the CPU the generated tokens
are the same in every case below. Neither is a guarantee for other arithmetic (bfloat16 can pick a
different token; D-030), and none of these tests claims it."""

import pytest
import torch

from scratchllm.model.cache import KVCache
from scratchllm.model.generate import generate
from scratchllm.model.gpt import GPT, GPTConfig

SMALL = dict(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4)
VARIANTS = {
    "gpt2": {},
    "gpt2-fast": {"fast_attention": True},
    "rope": {"position": "rope"},
    "modern": {"norm": "rmsnorm", "position": "rope", "ffn": "swiglu", "fast_attention": True},
}


def build(variant: str, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(**SMALL, **VARIANTS[variant])).eval()


def text(batch: int = 2, length: int = 12, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, 50, (batch, length), generator=torch.Generator().manual_seed(seed))


# ---------- the logits do not change ----------


@pytest.mark.parametrize("variant", VARIANTS)
def test_feeding_a_text_in_pieces_gives_the_logits_of_feeding_it_whole(variant: str):
    model, ids = build(variant), text(length=14)
    with torch.no_grad():
        whole = model(ids)[0]
        cache = model.new_cache()
        # a prompt of 5, then a chunk of 4 (several positions after stored ones), then single tokens
        pieces = [model(ids[:, :5], cache=cache)[0], model(ids[:, 5:9], cache=cache)[0]]
        pieces += [model(ids[:, i : i + 1], cache=cache)[0] for i in range(9, 14)]
    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, atol=1e-5, rtol=1e-5)
    assert cache.length == 14


@pytest.mark.parametrize("variant", VARIANTS)
def test_one_token_at_a_time_from_the_start_matches_too(variant: str):
    model, ids = build(variant), text(length=16)  # exactly the window
    with torch.no_grad():
        whole = model(ids)[0]
        cache = model.new_cache()
        steps = [model(ids[:, i : i + 1], cache=cache)[0] for i in range(16)]
    torch.testing.assert_close(torch.cat(steps, dim=1), whole, atol=1e-5, rtol=1e-5)


def test_the_batch_rows_do_not_leak_into_each_other():
    model, ids = build("modern"), text(batch=3)
    with torch.no_grad():
        both = model.new_cache()
        model(ids, cache=both)
        alone = model.new_cache()
        model(ids[1:2], cache=alone)
        nxt = text(batch=3, length=1, seed=5)
        a = model(nxt, cache=both)[0][1:2]
        b = model(nxt[1:2], cache=alone)[0]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


# ---------- the generated text does not change ----------


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("prompt_length", [1, 5, 16, 22])
def test_greedy_text_is_identical_with_and_without_the_cache(variant: str, prompt_length: int):
    """Prompts shorter than, equal to and longer than the window; 20 new tokens always cross it
    for the short ones, so both the cached part and the fall-back to recomputation are covered."""
    model, ids = build(variant), text(length=prompt_length)
    plain = generate(model, ids, 20, temperature=0, use_cache=False)
    cached = generate(model, ids, 20, temperature=0, use_cache=True)
    assert torch.equal(plain, cached)


@pytest.mark.parametrize("variant", ["gpt2", "modern"])
def test_sampled_text_is_identical_for_the_same_seed(variant: str):
    model, ids = build(variant), text(length=4)

    def run(use_cache: bool) -> torch.Tensor:
        gen = torch.Generator().manual_seed(7)
        return generate(
            model, ids, 25, temperature=0.9, top_k=10, generator=gen, use_cache=use_cache
        )

    assert torch.equal(run(False), run(True))


def test_end_of_text_handling_is_unchanged():
    model, ids = build("modern"), text(length=3)
    plain = generate(model, ids, 12, temperature=0, eos_id=7, use_cache=False)
    cached = generate(model, ids, 12, temperature=0, eos_id=7, use_cache=True)
    assert torch.equal(plain, cached)


def test_the_cache_is_the_default_and_the_model_sees_one_new_token_per_step():
    model, ids = build("modern"), text(batch=1, length=4)
    seen: list[int] = []
    model.tok_emb.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].size(1)))
    generate(model, ids, 6, temperature=0)  # 4 + 6 = 10 < 16: never leaves the cache
    assert seen == [4, 1, 1, 1, 1, 1]
    seen.clear()
    generate(model, ids, 6, temperature=0, use_cache=False)
    assert seen == [4, 5, 6, 7, 8, 9]  # the whole text again at every step


def test_after_the_window_is_full_the_model_recomputes_the_window():
    model, ids = build("modern"), text(batch=1, length=14)
    seen: list[int] = []
    model.tok_emb.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].size(1)))
    generate(model, ids, 5, temperature=0)  # lengths 14, 15, 16 fit; then the window slides
    assert seen == [14, 1, 1, 16, 16]


def test_a_model_without_a_cache_still_works():
    from test_generate import FakeLM, counting  # a model that offers no new_cache()

    out = generate(FakeLM(counting), torch.tensor([[1]]), 3, temperature=0)
    assert out.tolist() == [[1, 2, 3, 4]]


# ---------- the cache object ----------


def test_the_cache_refuses_what_would_not_fit_or_make_sense():
    model, ids = build("gpt2"), text(length=10)
    with torch.no_grad():
        cache = model.new_cache()
        model(ids, cache=cache)
        with pytest.raises(ValueError, match="exceed the context"):
            model(text(length=7, seed=3), cache=cache)  # 10 + 7 > 16
        with pytest.raises(ValueError, match="loss"):
            model(ids[:, :2], ids[:, :2], cache=model.new_cache())
        with pytest.raises(ValueError, match="batch"):
            model(text(batch=3, length=1), cache=cache)
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        model(ids[:, :2], cache=model.new_cache())


def test_the_cache_stores_each_layer_separately_and_in_order():
    cache = KVCache(n_layers=2, context_length=4)
    k0, v0 = torch.ones(1, 1, 2, 3), torch.ones(1, 1, 2, 3) * 2
    k1, v1 = torch.ones(1, 1, 2, 3) * 3, torch.ones(1, 1, 2, 3) * 4
    cache.update(0, k0, v0)
    keys, values = cache.update(1, k1, v1)
    assert (
        keys.shape == (1, 1, 2, 3)
        and keys.unique().tolist() == [3.0]
        and values.unique().tolist() == [4.0]
    )
    cache.advance(2)
    more_k, more_v = cache.update(0, torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1, 3))
    assert more_k.shape == (1, 1, 3, 3) and more_k[:, :, :2].unique().tolist() == [1.0]
    assert more_k[:, :, 2].unique().tolist() == [0.0] and more_v[:, :, :2].unique().tolist() == [
        2.0
    ]
