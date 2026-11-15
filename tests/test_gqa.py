"""Grouped-query attention (`n_kv_heads`): what it is, that it changes nothing when off, and that
the cache, generation, the config and LoRA follow it."""

import pytest
import torch

from scratchllm.finetune.lora import (
    LoRALinear,
    add_lora,
    lora_parameter_count,
    merge_lora,
    trainable_parameters,
)
from scratchllm.model.attention import CausalSelfAttention
from scratchllm.model.generate import generate
from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.training.checkpoint import model_from_state
from scratchllm.training.config import RunConfig

SMALL = dict(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4)
D, H, HD = 32, 4, 8  # d_model, query heads, head size


def build(n_kv_heads: int | None, seed: int = 0, **extra) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(**SMALL, n_kv_heads=n_kv_heads, **extra)).eval()


def ids(batch: int = 2, length: int = 12, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, 50, (batch, length), generator=torch.Generator().manual_seed(seed))


def as_multi_head(gqa: CausalSelfAttention) -> CausalSelfAttention:
    """The multi-head attention that GQA claims to be (no biases): each key/value head copied to
    every query head of its group, all other weights the same."""
    mha = CausalSelfAttention(D, H, 16, fast=gqa.fast, rope=gqa.rope).eval()
    w = gqa.qkv.weight
    q, k, v = w.split((D, gqa.kv_dim, gqa.kv_dim))

    def repeat(t: torch.Tensor) -> torch.Tensor:  # (kv_heads * hd, D) -> (H * hd, D)
        return t.view(gqa.n_kv_heads, HD, D).repeat_interleave(gqa.groups, dim=0).reshape(D, D)

    with torch.no_grad():
        mha.qkv.weight.copy_(torch.cat([q, repeat(k), repeat(v)]))
        mha.out_proj.weight.copy_(gqa.out_proj.weight)
    return mha


# ---------- off by default: nothing changes ----------


def test_without_the_option_the_model_and_its_checkpoints_are_unchanged():
    old_style = GPT(GPTConfig(**SMALL))  # a config made before the option existed
    explicit = build(n_kv_heads=4)
    assert old_style.blocks[0].attn.qkv.weight.shape == (3 * D, D)
    assert {k: v.shape for k, v in old_style.state_dict().items()} == {
        k: v.shape for k, v in explicit.state_dict().items()
    }
    config = GPTConfig(**SMALL).to_dict()
    del config["n_kv_heads"]  # an older checkpoint's config lacks the key
    model_from_state({"config": {"model": config}, "model": old_style.state_dict()})


# ---------- what GQA is ----------


@pytest.mark.parametrize("kv", [1, 2])
@pytest.mark.parametrize("variant", [{}, {"fast": True}, {"rope": True, "fast": True}])
def test_gqa_equals_multi_head_attention_with_shared_key_value_heads_repeated(kv, variant):
    torch.manual_seed(0)
    gqa = CausalSelfAttention(D, H, 16, n_kv_heads=kv, **variant).eval()
    x = torch.randn(2, 10, D)
    with torch.no_grad():
        torch.testing.assert_close(gqa(x), as_multi_head(gqa)(x), atol=1e-6, rtol=1e-5)


def test_heads_of_a_group_share_one_key_head_and_the_weights_show_it():
    torch.manual_seed(0)
    gqa = CausalSelfAttention(D, H, 16, n_kv_heads=2).eval()
    with torch.no_grad():
        _, weights = gqa.forward_with_weights(torch.randn(1, 6, D))
    assert weights.shape == (1, H, 6, 6)
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, H, 6))
    assert torch.equal(weights.triu(1), torch.zeros_like(weights))  # still causal


@pytest.mark.parametrize("kv", [1, 2, 4])
def test_the_key_value_projection_shrinks_by_the_group_factor(kv):
    attn = CausalSelfAttention(D, H, 16, n_kv_heads=kv)
    assert attn.kv_dim == kv * HD and attn.groups == H // kv
    assert attn.qkv.weight.shape == (D + 2 * kv * HD, D)


def test_the_fused_kernel_and_the_written_out_attention_agree():
    torch.manual_seed(0)
    slow = CausalSelfAttention(D, H, 16, n_kv_heads=2, rope=True).eval()
    fast = CausalSelfAttention(D, H, 16, n_kv_heads=2, rope=True, fast=True).eval()
    fast.load_state_dict(slow.state_dict())
    x = torch.randn(2, 9, D)
    with torch.no_grad():
        torch.testing.assert_close(fast(x), slow(x), atol=1e-6, rtol=1e-5)


def test_changing_the_future_never_changes_the_past():
    model, x = build(n_kv_heads=2, position="rope"), ids(length=10)
    y = x.clone()
    y[:, 6:] = (y[:, 6:] + 1) % 50
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0][:, :6], model(y)[0][:, :6])


@pytest.mark.parametrize("kv", [0, 3, 5])
def test_a_head_count_that_does_not_divide_is_refused(kv):
    with pytest.raises(ValueError, match="multiple of n_kv_heads"):
        CausalSelfAttention(D, H, 16, n_kv_heads=kv)


def test_the_run_config_refuses_a_bad_count():
    base = {
        "name": "x",
        "model": {"n_heads": 6, "n_kv_heads": 4},
        "data": {"tokens_dir": "t"},
        "optim": {"lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 1},
        "train": {"max_steps": 2, "micro_batch_size": 1, "eval_interval": 1,
                  "checkpoint_interval": 1},
    }  # fmt: skip
    with pytest.raises(ValueError, match="multiple of n_kv_heads"):
        RunConfig.model_validate(base)
    base["model"]["n_kv_heads"] = 2
    assert RunConfig.model_validate(base).model.to_gpt().n_kv_heads == 2


# ---------- the cache and generation ----------


@pytest.mark.parametrize("kv", [1, 2])
def test_feeding_in_pieces_with_the_cache_gives_the_logits_of_feeding_whole(kv):
    model, x = build(n_kv_heads=kv, position="rope", fast_attention=True), ids(length=14)
    with torch.no_grad():
        whole = model(x)[0]
        cache = model.new_cache()
        parts = [model(x[:, a:b], cache=cache)[0] for a, b in ((0, 5), (5, 6), (6, 14))]
    torch.testing.assert_close(torch.cat(parts, dim=1), whole, atol=1e-5, rtol=1e-5)


def test_the_cache_holds_only_the_shared_heads():
    sizes = {}
    for kv in (4, 2, 1):
        model = build(n_kv_heads=kv)
        cache = model.new_cache()
        with torch.no_grad():
            model(ids(), cache=cache)
        sizes[kv] = cache.nbytes()
    assert sizes[4] == 2 * sizes[2] == 4 * sizes[1]


def test_generation_with_and_without_the_cache_agrees():
    model = build(n_kv_heads=2, position="rope", ffn="swiglu")
    prompt = ids(batch=1, length=4)
    with torch.no_grad():
        cached = generate(model, prompt, 10, temperature=0.0)
        plain = generate(model, prompt, 10, temperature=0.0, use_cache=False)
    assert torch.equal(cached, plain)


def test_a_gqa_model_trains():
    model = build(n_kv_heads=2).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = ids(length=13)
    first = None
    for _ in range(30):
        _, loss = model(x[:, :-1], x[:, 1:])
        assert loss is not None
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * first


# ---------- LoRA ----------


def test_lora_slices_follow_the_narrower_keys_and_values():
    model = build(n_kv_heads=2)
    add_lora(model, rank=4, alpha=4, targets=("attn.q", "attn.k", "attn.v"))
    layer = model.blocks[0].attn.qkv
    assert isinstance(layer, LoRALinear)
    assert layer.slices == [(0, 32), (32, 48), (48, 64)]
    expected = lora_parameter_count(D, model.cfg.hidden, 2, 4, ("attn.q", "attn.k", "attn.v"), 16)
    assert trainable_parameters(model) == expected


def test_lora_on_a_gqa_model_merges_to_the_same_outputs():
    model = build(n_kv_heads=2, position="rope")
    add_lora(model, rank=4, alpha=8, targets=("attn.q", "attn.v"))
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_b." in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        before = model(ids())[0]
        merge_lora(model)
        torch.testing.assert_close(model(ids())[0], before, atol=1e-5, rtol=1e-5)
