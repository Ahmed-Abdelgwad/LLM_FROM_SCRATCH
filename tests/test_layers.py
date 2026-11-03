import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from scratchllm.model.layers import GELU, FeedForward, LayerNorm


def rand(*shape: int, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale


# ---------- layer norm ----------


def test_layer_norm_matches_the_library_with_random_scale_and_shift():
    ours, ref = LayerNorm(16), torch.nn.LayerNorm(16)
    with torch.no_grad():
        for layer in (ours, ref):
            layer.weight.copy_(rand(16, seed=1))
            layer.bias.copy_(rand(16, seed=2))
    x = rand(4, 7, 16, seed=3, scale=5.0) + 3.0
    torch.testing.assert_close(ours(x), ref(x), atol=1e-5, rtol=1e-5)


def test_each_token_is_normalized_over_its_own_numbers():
    out = LayerNorm(64)(rand(3, 5, 64, scale=7.0) + 11.0)
    torch.testing.assert_close(out.mean(-1), torch.zeros(3, 5), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.var(-1, unbiased=False), torch.ones(3, 5), atol=1e-3, rtol=0)


def test_a_token_does_not_influence_the_normalization_of_another():
    norm = LayerNorm(16)
    x = rand(2, 4, 16)
    changed = x.clone()
    changed[:, 2] *= 100
    assert torch.equal(norm(x)[:, [0, 1, 3]], norm(changed)[:, [0, 1, 3]])


def test_a_constant_vector_gives_zeros_not_nan():
    out = LayerNorm(8)(torch.full((1, 1, 8), 3.0))
    assert torch.all(out == 0) and torch.isfinite(out).all()


def test_without_bias_there_is_no_shift_parameter():
    assert [n for n, _ in LayerNorm(8, bias=False).named_parameters()] == ["weight"]
    assert [n for n, _ in LayerNorm(8).named_parameters()] == ["weight", "bias"]


def test_bfloat16_input_comes_back_as_bfloat16_and_close_to_float32():
    x = rand(2, 3, 32)
    out = LayerNorm(32)(x.bfloat16())
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), LayerNorm(32)(x), atol=0.05, rtol=0.05)


# ---------- GELU ----------


def test_gelu_matches_the_library_tanh_version():
    x = torch.linspace(-8, 8, 1001)
    torch.testing.assert_close(GELU()(x), F.gelu(x, approximate="tanh"), atol=1e-6, rtol=1e-5)


def test_gelu_shape_of_the_curve():
    g = GELU()
    assert g(torch.tensor(0.0)).item() == 0.0
    torch.testing.assert_close(g(torch.tensor(8.0)), torch.tensor(8.0), atol=1e-4, rtol=0)
    assert abs(g(torch.tensor(-8.0)).item()) < 1e-4
    assert g(torch.tensor(-0.5)).item() < 0  # unlike ReLU, a small negative dip


# ---------- feed-forward ----------


def test_feed_forward_shape_and_parameter_count():
    ff = FeedForward(16, 64)
    assert ff(rand(2, 5, 16)).shape == (2, 5, 16)
    assert sum(p.numel() for p in ff.parameters()) == 2 * 16 * 64
    assert (
        sum(p.numel() for p in FeedForward(16, 64, bias=True).parameters()) == 2 * 16 * 64 + 64 + 16
    )


def test_feed_forward_treats_every_token_separately():
    ff = FeedForward(16, 64)
    x = rand(2, 5, 16)
    torch.testing.assert_close(ff(x)[:, 3], ff(x[:, 3:4])[:, 0])


def test_feed_forward_dropout_only_in_training():
    ff = FeedForward(16, 64, dropout=0.5).eval()
    x = rand(2, 5, 16)
    assert torch.equal(ff(x), ff(x))
    ff.train()
    assert not torch.equal(ff(x), ff(x))


@pytest.mark.parametrize("d_ff", [1, 16, 100])
def test_feed_forward_any_width(d_ff: int):
    assert FeedForward(8, d_ff)(rand(1, 2, 8)).shape == (1, 2, 8)
