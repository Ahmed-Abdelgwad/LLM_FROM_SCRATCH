"""LoRA (finetune/lora.py): the contract of the paper's update and of merging it."""

import pytest
import torch
from torch import nn

from scratchllm.finetune.lora import (
    PAPER_TARGETS,
    LoRALinear,
    add_lora,
    lora_parameter_count,
    lora_state,
    merge_lora,
    trainable_parameters,
)
from scratchllm.model.generate import generate
from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.training.checkpoint import model_from_state

SMALL = dict(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4)
ALL = ("attn.q", "attn.k", "attn.v", "attn.out_proj", "ff.up", "ff.gate", "ff.down")


def build(seed: int = 0, **extra) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(**SMALL, **extra))


def ids(seed: int = 1) -> torch.Tensor:
    return torch.randint(0, 50, (2, 12), generator=torch.Generator().manual_seed(seed))


def randomize_b(model: nn.Module, seed: int = 3) -> None:
    """Give every B a non-zero value, as training would."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_b." in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)


# ---------- at initialization the model is the base model ----------


@pytest.mark.parametrize("modern", [False, True])
def test_at_initialization_the_output_is_the_base_models_bit_for_bit(modern: bool):
    extra = {"position": "rope", "ffn": "swiglu"} if modern else {}
    base, adapted = build(**extra).eval(), build(**extra).eval()
    add_lora(adapted, rank=4, alpha=4, targets=ALL if modern else ("attn.q", "attn.v", "ff.up"))
    assert torch.equal(base(ids())[0], adapted(ids())[0])


# ---------- only the adapters learn ----------


def test_only_a_and_b_are_trainable_and_they_receive_gradients():
    model = build()
    add_lora(model, rank=4, alpha=8)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_a." in n or ".lora_b." in n for n in trainable)
    _, loss = model(ids(), ids(2))
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert all(grads[n] is not None for n in trainable)
    assert all(grads[n] is None for n in grads if n not in trainable)
    # with B = 0 the gradient reaches B (through A x) but not A (through B)
    assert any(grads[n].abs().sum() > 0 for n in trainable if ".lora_b." in n)
    assert all(grads[n].abs().sum() == 0 for n in trainable if ".lora_a." in n)


def test_a_lora_layer_used_on_its_own_freezes_its_base():
    layer = LoRALinear(nn.Linear(6, 4), rank=2, alpha=2)
    assert not any(p.requires_grad for p in layer.base.parameters())
    assert all(p.requires_grad for p in [*layer.lora_a, *layer.lora_b])


def test_an_optimizer_step_leaves_every_base_weight_unchanged():
    model = build()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    add_lora(model, rank=4, alpha=8, targets=ALL[:4] + ("ff.up", "ff.down"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        _, loss = model(ids(), ids(2))
        loss.backward()
        opt.step()
        opt.zero_grad()
    merged_names = {k.replace(".base.", "."): v for k, v in model.state_dict().items()}
    for key, value in before.items():
        assert torch.equal(merged_names[key], value), key


# ---------- the fused query/key/value map ----------


def test_q_and_v_adapters_leave_the_key_third_of_qkv_exactly_as_it_was():
    model = build()
    add_lora(model, rank=4, alpha=4, targets=PAPER_TARGETS)
    randomize_b(model)
    layer = model.blocks[0].attn.qkv
    assert isinstance(layer, LoRALinear) and layer.slices == [(0, 32), (64, 96)]
    x = torch.randn(3, 5, 32)
    out, plain = layer(x), layer.base(x)
    assert torch.equal(out[..., 32:64], plain[..., 32:64])  # keys untouched
    assert not torch.allclose(out[..., :32], plain[..., :32])  # queries changed
    assert not torch.allclose(out[..., 64:], plain[..., 64:])  # values changed


def test_parameter_count_matches_the_formula_r_times_in_plus_out():
    for targets in (PAPER_TARGETS, ALL):
        model = build(ffn="swiglu")
        add_lora(model, rank=8, alpha=8, targets=targets)
        expected = lora_parameter_count(32, model.cfg.hidden, 2, 8, targets)
        assert trainable_parameters(model) == expected
    model = build()  # GELU feed-forward: no gate
    with pytest.raises(ValueError, match="ff.gate"):
        add_lora(model, rank=8, alpha=8, targets=("ff.gate",))


def test_unknown_or_misspelled_targets_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        add_lora(build(), rank=4, alpha=4, targets=("attn.query",))
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(4, 4), rank=0, alpha=1)
    with pytest.raises(ValueError, match="slice"):
        LoRALinear(nn.Linear(4, 6), rank=1, alpha=1, slices=[(0, 4), (2, 6)])


def test_the_scale_is_alpha_over_rank():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(6, 5, bias=False), rank=2, alpha=6)
    with torch.no_grad():
        layer.lora_b[0].fill_(1.0)
    x = torch.randn(4, 6)
    expected = layer.base(x) + 3.0 * (x @ layer.lora_a[0].T @ layer.lora_b[0].T)
    torch.testing.assert_close(layer(x), expected)


# ---------- merging ----------


@pytest.mark.parametrize("bias", [False, True])
def test_merging_gives_the_same_outputs_and_a_plain_gpt_again(bias: bool):
    model = build(bias=bias, position="rope", ffn="swiglu").eval()
    plain_keys = set(model.state_dict())
    add_lora(model, rank=4, alpha=8, targets=ALL)
    randomize_b(model)
    with torch.no_grad():
        before = model(ids())[0]
    assert merge_lora(model) == 2 * 5  # per layer: qkv, out_proj, gate, up, down
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert set(model.state_dict()) == plain_keys
    with torch.no_grad():
        torch.testing.assert_close(model(ids())[0], before, atol=1e-5, rtol=1e-5)
    # and the ordinary checkpoint loader builds it
    state = {"config": {"model": model.cfg.to_dict()}, "model": model.state_dict()}
    reloaded = model_from_state(state)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids())[0], before, atol=1e-5, rtol=1e-5)


def test_the_adapter_state_holds_only_a_and_b():
    model = build()
    add_lora(model, rank=2, alpha=2)
    state = lora_state(model)
    assert state and all(".lora_a." in k or ".lora_b." in k for k in state)
    assert sum(v.numel() for v in state.values()) == trainable_parameters(model)


def test_generation_with_the_kv_cache_works_through_the_adapters():
    model = build(position="rope", ffn="swiglu", fast_attention=True).eval()
    add_lora(model, rank=4, alpha=8, targets=ALL)
    randomize_b(model)
    prompt = ids()[:, :5]
    plain = generate(model, prompt, 15, temperature=0, use_cache=False)
    cached = generate(model, prompt, 15, temperature=0, use_cache=True)
    assert torch.equal(plain, cached)
