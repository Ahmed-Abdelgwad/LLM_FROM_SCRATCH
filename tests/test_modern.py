"""The "modern" options of D-027: RMS norm, SwiGLU feed-forward, rotary position embeddings.

Each is checked against what it is defined as (the library version, a hand formula, or the
property that is the reason for it), and the defaults are checked to still be GPT-2.
"""

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from scratchllm.model.attention import CausalSelfAttention, apply_rotary, rotary_cos_sin
from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.model.layers import FeedForward, RMSNorm, SwiGLUFeedForward, swiglu_hidden


def rand(*shape: int, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale


SMALL = dict(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4)
MODERN = dict(norm="rmsnorm", position="rope", ffn="swiglu")


def tokens(batch: int = 3, length: int = 12, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, 50, (batch, length), generator=torch.Generator().manual_seed(seed))


# ---------- RMS norm ----------


def test_rms_norm_matches_the_library_with_a_random_scale():
    ours, ref = RMSNorm(16, eps=1e-6), torch.nn.RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        for layer in (ours, ref):
            layer.weight.copy_(rand(16, seed=1))
    x = rand(4, 7, 16, seed=3, scale=5.0) + 3.0
    torch.testing.assert_close(ours(x), ref(x), atol=1e-5, rtol=1e-5)


def test_rms_norm_gives_unit_root_mean_square_and_keeps_the_mean():
    x = rand(3, 5, 64, scale=7.0) + 11.0
    out = RMSNorm(64)(x)
    torch.testing.assert_close(out.pow(2).mean(-1).sqrt(), torch.ones(3, 5), atol=1e-4, rtol=0)
    assert out.mean(-1).abs().min() > 0.5  # unlike LayerNorm it does not re-center


def test_rms_norm_has_only_a_scale_and_works_in_bfloat16():
    assert [n for n, _ in RMSNorm(8).named_parameters()] == ["weight"]
    x = rand(2, 3, 32, seed=4)
    out = RMSNorm(32)(x.bfloat16())
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), RMSNorm(32)(x), atol=0.05, rtol=0.05)


# ---------- SwiGLU ----------


def test_swiglu_is_down_of_silu_gate_times_up():
    ff = SwiGLUFeedForward(8, 20)
    x = rand(2, 5, 8)
    expected = ff.down(F.silu(ff.gate(x)) * ff.up(x))
    assert torch.equal(ff(x), expected)
    assert ff(x).shape == x.shape


def test_swiglu_gate_closed_means_nothing_passes():
    ff = SwiGLUFeedForward(8, 20)
    with torch.no_grad():
        ff.gate.weight.zero_()  # silu(0) = 0
    assert torch.count_nonzero(ff(rand(2, 5, 8))) == 0


@pytest.mark.parametrize("d_model", [384, 768, 1024])
def test_swiglu_width_keeps_the_parameters_of_the_4x_gelu_layer(d_model: int):
    """Rounding the width to a multiple of 64 costs a few percent only for very small d_model
    (64 would be 12% over), so the claim is checked at the sizes this project uses."""
    gelu = sum(p.numel() for p in FeedForward(d_model, 4 * d_model).parameters())
    swiglu = sum(p.numel() for p in SwiGLUFeedForward(d_model, swiglu_hidden(d_model)).parameters())
    assert abs(swiglu - gelu) / gelu < 0.02


def test_at_the_project_size_swiglu_has_exactly_the_parameters_of_the_gelu_layer():
    assert swiglu_hidden(384) == 1024
    gelu = sum(p.numel() for p in FeedForward(384, 1536).parameters())
    swiglu = sum(p.numel() for p in SwiGLUFeedForward(384, 1024).parameters())
    assert swiglu == gelu == 1_179_648


# ---------- rotary position embeddings ----------


def test_rotation_keeps_every_vector_length_and_leaves_position_zero_alone():
    cos, sin = rotary_cos_sin(16, 8)
    x = rand(2, 3, 16, 8)
    out = apply_rotary(x, cos, sin)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[:, :, 0], x[:, :, 0])  # angle 0
    assert not torch.allclose(out[:, :, 5], x[:, :, 5])


def test_the_first_pair_turns_by_one_radian_per_position():
    cos, sin = rotary_cos_sin(8, 4, theta=10000.0)
    assert torch.allclose(cos[:, 0], torch.arange(8.0).cos(), atol=1e-6)
    assert torch.allclose(sin[:, 0], torch.arange(8.0).sin(), atol=1e-6)
    # the pair (0, 2) shares its angle, so both halves use the same table entry
    assert torch.equal(cos[:, 0], cos[:, 2]) and torch.equal(sin[:, 1], sin[:, 3])


def test_the_rotation_is_the_documented_one_by_hand():
    """(a, b) -> (a cos t - b sin t, b cos t + a sin t) for the pair (i, i + head_dim/2). The
    opposite direction would also be a consistent rotary embedding, so this pins the convention."""
    cos, sin = rotary_cos_sin(4, 4, theta=100.0)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).expand(1, 4, 4)  # the same vector at 4 positions
    out = apply_rotary(x, cos, sin)[0, 2]  # position 2
    t0, t1 = 2.0, 2.0 * 100.0 ** (-2 / 4)  # angles of pair 0 (dims 0,2) and pair 1 (dims 1,3)
    expected = torch.tensor([
        1.0 * math.cos(t0) - 3.0 * math.sin(t0),
        2.0 * math.cos(t1) - 4.0 * math.sin(t1),
        3.0 * math.cos(t0) + 1.0 * math.sin(t0),
        4.0 * math.cos(t1) + 2.0 * math.sin(t1),
    ])  # fmt: skip
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shift", [1, 5, 9])
def test_the_score_between_two_positions_depends_only_on_their_distance(shift: int):
    """The reason for rotary embeddings: q_i . k_j is a function of i - j, not of i and j."""
    cos, sin = rotary_cos_sin(32, 16)
    q, k = rand(16, seed=1), rand(16, seed=2)

    def score(i: int, j: int) -> torch.Tensor:
        qi = apply_rotary(q[None, None], cos[i : i + 1], sin[i : i + 1])
        kj = apply_rotary(k[None, None], cos[j : j + 1], sin[j : j + 1])
        return (qi * kj).sum()

    for i, j in [(4, 1), (6, 6), (10, 3)]:
        torch.testing.assert_close(score(i, j), score(i + shift, j + shift), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(score(4, 1), score(4, 2), atol=1e-4)  # distance does matter


def test_an_odd_head_size_is_refused():
    with pytest.raises(ValueError, match="even"):
        rotary_cos_sin(8, 5)


def rope_attention(fast: bool) -> CausalSelfAttention:
    torch.manual_seed(0)
    return CausalSelfAttention(32, 4, 16, fast=fast, rope=True).eval()


def test_rotary_attention_fast_path_matches_the_explicit_one():
    ref, fast = rope_attention(False), rope_attention(True)
    fast.load_state_dict(ref.state_dict())
    x = rand(2, 12, 32, seed=7)
    torch.testing.assert_close(fast(x), ref(x), atol=1e-5, rtol=1e-5)


def test_rotary_attention_is_still_causal():
    attn = rope_attention(False)
    x = rand(1, 12, 32, seed=8)
    y = x.clone()
    y[:, 7:] = rand(1, 5, 32, seed=9)  # change the future of position 6
    torch.testing.assert_close(attn(x)[:, :7], attn(y)[:, :7])
    assert not torch.allclose(attn(x)[:, 7:], attn(y)[:, 7:])


def test_rotary_tables_are_not_saved_in_a_checkpoint():
    assert not any("rope" in k for k in rope_attention(False).state_dict())


def test_shifting_the_whole_text_changes_nothing_in_a_rotary_attention_layer():
    """Relative positions: the same tokens at positions 0..7 or 3..10 give the same outputs
    (for a layer, with no absolute position anywhere). Compared on the last 8 of 11 inputs
    whose first 3 are dropped: with a causal mask the dropped ones would be seen by the later
    ones, so instead compare scores directly through the attention weights of two windows."""
    attn = rope_attention(False)
    x = rand(1, 8, 32, seed=11)
    w_start = attn.forward_with_weights(x)[1]
    # the same 8 vectors read as if they came 3 places later: rotate by the table shifted by 3
    cos, sin = attn.rope_cos, attn.rope_sin
    q, k, _ = (attn._heads(t) for t in attn.qkv(x).split(32, dim=2))
    q = apply_rotary(q, cos[3:11], sin[3:11])
    k = apply_rotary(k, cos[3:11], sin[3:11])
    scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
    w_shifted = torch.softmax(scores.masked_fill(attn.future[:8, :8], float("-inf")), dim=-1)
    torch.testing.assert_close(w_shifted, w_start, atol=1e-5, rtol=1e-5)


# ---------- the whole model ----------


def test_the_defaults_are_still_gpt2():
    cfg = GPTConfig(**SMALL)
    assert (cfg.norm, cfg.position, cfg.ffn) == ("layernorm", "learned", "gelu")
    m = GPT(cfg)
    assert m.pos_emb is not None and isinstance(m.blocks[0].ff, FeedForward)
    assert hasattr(m.blocks[0].norm1, "bias") or m.blocks[0].norm1.bias is None


def test_the_17m_model_keeps_its_exact_parameter_count_and_names():
    cfg = GPTConfig(vocab_size=16384, context_length=512, d_model=384, n_layers=6, n_heads=6)
    m = GPT(cfg)
    assert m.num_parameters() == 17_109_888
    names = list(m.state_dict())
    assert "pos_emb.weight" in names and "blocks.0.ff.up.weight" in names
    assert "blocks.0.ff.gate.weight" not in names


def test_an_old_checkpoint_config_without_the_new_keys_builds_the_same_model():
    old = {
        k: v for k, v in GPTConfig(**SMALL).to_dict().items() if k not in MODERN | {"rope_theta": 0}
    }
    assert GPTConfig(**old) == GPTConfig(**SMALL)


def test_modern_model_shapes_loss_and_start_at_a_uniform_guess():
    cfg = GPTConfig(vocab_size=1000, context_length=32, d_model=64, n_layers=2, n_heads=4, **MODERN)
    m = GPT(cfg).eval()
    x = torch.randint(0, 1000, (8, 32), generator=torch.Generator().manual_seed(0))
    logits, loss = m(x, torch.roll(x, -1, dims=1))
    assert logits.shape == (8, 32, 1000) and loss is not None
    assert loss.item() == pytest.approx(math.log(1000), abs=0.15)


def test_rotary_model_has_no_position_table_and_swiglu_the_expected_matrices():
    m = GPT(GPTConfig(**SMALL, **MODERN))
    names = list(m.state_dict())
    assert m.pos_emb is None and not any(n.startswith("pos_emb") for n in names)
    assert {"blocks.0.ff.gate.weight", "blocks.0.ff.up.weight", "blocks.0.ff.down.weight"} <= set(
        names
    )
    assert "blocks.0.norm1.bias" not in names and "final_norm.bias" not in names
    base = GPT(GPTConfig(**SMALL))
    assert base.num_parameters() - m.num_parameters() >= SMALL["context_length"] * SMALL["d_model"]
    assert m.num_parameters(include_embeddings=False) > 0


def test_the_residual_output_maps_get_the_smaller_start_in_the_modern_model():
    cfg = GPTConfig(vocab_size=50, context_length=16, d_model=128, n_layers=4, n_heads=4, **MODERN)
    m = GPT(cfg)
    target = cfg.init_std / math.sqrt(2 * cfg.n_layers)
    for block in m.blocks:
        assert block.attn.out_proj.weight.std().item() == pytest.approx(target, rel=0.1)
        assert block.ff.down.weight.std().item() == pytest.approx(target, rel=0.1)  # type: ignore[union-attr]


def test_modern_model_is_causal_and_a_prefix_gives_the_same_numbers():
    m = GPT(GPTConfig(**SMALL, **MODERN)).eval()
    x = tokens(length=12)
    full = m(x)[0]
    torch.testing.assert_close(m(x[:, :7])[0], full[:, :7], atol=1e-5, rtol=1e-5)
    y = x.clone()
    y[:, 7:] = tokens(length=5, seed=5)
    torch.testing.assert_close(m(y)[0][:, :7], full[:, :7], atol=1e-5, rtol=1e-5)


def test_without_any_position_signal_a_rotary_model_still_tells_order_apart():
    """A learned-position model with zeroed positions and one layer cannot (test_gpt); rotary can,
    because the order is inside the attention scores."""
    cfg = GPTConfig(
        vocab_size=50, context_length=16, d_model=32, n_layers=1, n_heads=4, position="rope"
    )
    m = GPT(cfg).eval()
    a = torch.tensor([[3, 7, 11, 5]])
    b = torch.tensor([[7, 3, 11, 5]])  # the first two swapped; the last token is the same
    assert not torch.allclose(m(a)[0][:, -1], m(b)[0][:, -1], atol=1e-6)


def test_every_parameter_of_the_modern_model_gets_a_gradient():
    m = GPT(GPTConfig(**SMALL, **MODERN))
    _, loss = m(tokens(), tokens(seed=2))
    assert loss is not None
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert missing == []


def test_the_modern_model_trains_and_generates_through_the_normal_paths(make_cfg):
    from scratchllm.model.generate import generate
    from scratchllm.training.checkpoint import load_model
    from scratchllm.training.trainer import train

    cfg = make_cfg("modern", model=MODERN, train={"max_steps": 30, "eval_interval": 30})
    train(cfg)
    model = load_model(cfg.run_dir / "last.pt")
    assert model.cfg.position == "rope" and model.pos_emb is None
    out = generate(model, torch.tensor([1, 2, 3]), 5, temperature=0)
    assert out.shape == (8,)
