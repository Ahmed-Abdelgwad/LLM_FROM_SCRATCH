import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from scratchllm.model.attention import CausalSelfAttention

D, H, T, B = 32, 4, 10, 3


def make(dropout: float = 0.0, bias: bool = False, d: int = D, h: int = H) -> CausalSelfAttention:
    torch.manual_seed(0)
    return CausalSelfAttention(d, h, context_length=16, dropout=dropout, bias=bias).eval()


def sample(batch: int = B, length: int = T, d: int = D, seed: int = 1) -> torch.Tensor:
    return torch.randn(batch, length, d, generator=torch.Generator().manual_seed(seed))


def reference_with_sdpa(att: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """The same layer computed by the library's attention kernel."""
    q, k, v = att.qkv(x).split(att.d_model, dim=2)
    b, t, c = x.shape

    def heads(z: torch.Tensor) -> torch.Tensor:
        return z.view(b, t, att.n_heads, att.head_dim).transpose(1, 2)

    out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), is_causal=True)
    return att.out_proj(out.transpose(1, 2).reshape(b, t, c))


def reference_loop(att: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """The same layer with explicit loops over examples, heads and positions: slow and plain."""
    b, t, c = x.shape
    hd = att.head_dim
    qkv = att.qkv(x)
    out = torch.zeros(b, t, c)
    for n in range(b):
        for head in range(att.n_heads):
            lo = head * hd
            q = qkv[n, :, lo : lo + hd]
            k = qkv[n, :, c + lo : c + lo + hd]
            v = qkv[n, :, 2 * c + lo : 2 * c + lo + hd]
            for i in range(t):
                scores = torch.stack([q[i] @ k[j] / math.sqrt(hd) for j in range(i + 1)])
                weights = torch.softmax(scores, dim=0)
                out[n, i, lo : lo + hd] = sum(weights[j] * v[j] for j in range(i + 1))
    return att.out_proj(out)


# ---------- shapes and the numbers ----------


def test_output_has_the_shape_of_the_input():
    assert make()(sample()).shape == (B, T, D)


def test_matches_the_library_attention_kernel():
    att, x = make(), sample()
    torch.testing.assert_close(att(x), reference_with_sdpa(att, x), atol=1e-5, rtol=1e-5)


def test_matches_a_plain_loop_over_heads_and_positions():
    att, x = make(bias=True), sample(batch=2, length=6)
    torch.testing.assert_close(att(x), reference_loop(att, x), atol=1e-5, rtol=1e-5)


def test_the_fused_matrix_is_three_matrices_side_by_side():
    att, x = make(), sample()
    wq, wk, wv = att.qkv.weight.split(D, dim=0)
    q, k, v = att.qkv(x).split(D, dim=2)
    torch.testing.assert_close(q, x @ wq.T)
    torch.testing.assert_close(k, x @ wk.T)
    torch.testing.assert_close(v, x @ wv.T)


# ---------- the weights ----------


def test_weights_are_a_distribution_over_the_past_and_nothing_over_the_future():
    _, w = make().forward_with_weights(sample())
    assert w.shape == (B, H, T, T)
    torch.testing.assert_close(w.sum(-1), torch.ones(B, H, T))
    assert torch.all(w >= 0)
    assert torch.all(torch.triu(w, diagonal=1) == 0)  # exactly zero, not just small


def test_the_first_token_can_only_look_at_itself():
    _, w = make().forward_with_weights(sample())
    assert torch.all(w[:, :, 0, 0] == 1)


def test_a_token_may_attend_to_itself():
    # the mask removes j > i, not j == i
    _, w = make().forward_with_weights(sample())
    assert torch.all(torch.diagonal(w, dim1=-2, dim2=-1) > 0)


def test_the_scores_are_scaled_by_the_head_size_not_the_model_size():
    att, x = make(), sample(length=4)
    q, k, _ = att.qkv(x).split(D, dim=2)
    qh = q.view(B, 4, H, D // H).transpose(1, 2)
    kh = k.view(B, 4, H, D // H).transpose(1, 2)
    unmasked = torch.softmax(qh @ kh.transpose(-2, -1) / math.sqrt(D // H), dim=-1)
    _, w = att.forward_with_weights(x)
    # row 3 sees all four positions, so it equals the unmasked softmax
    torch.testing.assert_close(w[:, :, 3], unmasked[:, :, 3])


# ---------- causality: the property the whole layer exists for ----------


@pytest.mark.parametrize("cut", [0, 3, 8])
def test_changing_the_future_never_changes_the_past(cut: int):
    att, x = make(), sample()
    changed = x.clone()
    changed[:, cut + 1 :] = torch.randn_like(changed[:, cut + 1 :]) * 10
    torch.testing.assert_close(att(x)[:, : cut + 1], att(changed)[:, : cut + 1], atol=1e-6, rtol=0)
    if cut + 1 < T:
        assert not torch.allclose(att(x)[:, cut + 1 :], att(changed)[:, cut + 1 :])


def test_a_shorter_sequence_gives_the_same_numbers_as_the_start_of_a_longer_one():
    att, x = make(), sample()
    torch.testing.assert_close(att(x[:, :5]), att(x)[:, :5], atol=1e-6, rtol=0)


# ---------- training behaviour ----------


def test_dropout_acts_in_training_only():
    att, x = make(dropout=0.5), sample()
    assert torch.equal(att(x), att(x))  # eval mode: deterministic
    att.train()
    torch.manual_seed(0)
    first = att(x)
    assert not torch.equal(first, att(x))


def test_every_parameter_receives_a_gradient():
    att = make(bias=True).train()
    att(sample()).square().sum().backward()
    for name, p in att.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_parameter_count():
    assert sum(p.numel() for p in make().parameters()) == 4 * D * D
    assert sum(p.numel() for p in make(bias=True).parameters()) == 4 * D * D + 3 * D + D


def test_the_mask_is_not_saved_in_a_checkpoint():
    assert list(make().state_dict()) == ["qkv.weight", "out_proj.weight"]


# ---------- refused inputs ----------


def test_wrong_sizes_are_refused():
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(30, 4, 16)
    att = make()
    with pytest.raises(ValueError, match="context"):
        att(sample(length=17))
    with pytest.raises(ValueError, match="last dimension"):
        att(sample(d=D + 1))


# ---------- the fast path gives the same layer ----------


def fast_pair(**kwargs) -> tuple[CausalSelfAttention, CausalSelfAttention]:
    slow = make(**kwargs)
    fast = CausalSelfAttention(D, H, 16, fast=True, **{k: v for k, v in kwargs.items() if k != "h"})
    fast.load_state_dict(slow.state_dict())
    return slow, fast.eval()


def test_the_fast_path_gives_the_same_numbers_as_the_explicit_one():
    slow, fast = fast_pair(bias=True)
    x = sample()
    torch.testing.assert_close(fast(x), slow(x), atol=1e-5, rtol=1e-5)


def test_the_fast_path_is_causal_too():
    _, fast = fast_pair()
    x = sample()
    changed = x.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 10
    torch.testing.assert_close(fast(x)[:, :5], fast(changed)[:, :5], atol=1e-6, rtol=0)


def test_the_fast_path_trains_and_drops_out_only_in_training():
    _, fast = fast_pair(dropout=0.5)
    x = sample()
    assert torch.equal(fast(x), fast(x))
    fast.train()
    assert not torch.equal(fast(x), fast(x))
    fast(x).square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in fast.parameters())


def test_the_fast_path_refuses_the_same_inputs():
    _, fast = fast_pair()
    with pytest.raises(ValueError, match="context"):
        fast(sample(length=17))
    with pytest.raises(ValueError, match="last dimension"):
        fast(sample(d=D + 1))
