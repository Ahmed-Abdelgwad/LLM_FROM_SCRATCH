import math

import pytest
import torch

from scratchllm.model.gpt import GPT, Block, GPTConfig

CFG = GPTConfig(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4)


def model(cfg: GPTConfig = CFG, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(cfg).eval()


def ids(batch: int = 3, length: int = 12, vocab: int = 50, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))


# ---------- shapes, loss, parameters ----------


def test_logits_have_one_score_per_vocabulary_item_at_every_position():
    logits, loss = model()(ids())
    assert logits.shape == (3, 12, 50) and loss is None


def test_loss_is_the_mean_cross_entropy_of_the_next_token():
    m, x = model(), ids()
    y = ids(seed=2)
    logits, loss = m(x, y)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 50), y.reshape(-1))
    assert loss is not None and torch.equal(loss, expected)


def test_an_untrained_model_is_as_good_as_a_uniform_guess():
    cfg = GPTConfig(vocab_size=1000, context_length=32, d_model=64, n_layers=2, n_heads=4)
    m = model(cfg)
    x = torch.randint(0, 1000, (8, 32), generator=torch.Generator().manual_seed(0))
    _, loss = m(x, torch.roll(x, -1, dims=1))
    assert loss is not None and abs(loss.item() - math.log(1000)) < 0.1


def test_parameter_count_follows_the_architecture():
    c = CFG
    d, h = c.d_model, c.hidden
    per_block = 4 * d * d + 2 * d * h + 2 * d  # attention, feed-forward, two norm scales
    expected = c.vocab_size * d + c.context_length * d + c.n_layers * per_block + d
    m = model()
    assert m.num_parameters() == expected
    assert (
        m.num_parameters(include_embeddings=False)
        == expected - (c.vocab_size + c.context_length) * d
    )


def test_the_default_local_model_is_about_seventeen_million_parameters():
    m = GPT(GPTConfig())
    assert 16.5e6 < m.num_parameters() < 17.5e6


def test_tied_embeddings_share_one_matrix_and_untied_cost_a_vocabulary_more():
    tied, untied = model(), model(GPTConfig(**{**CFG.to_dict(), "tie_embeddings": False}))
    assert tied.lm_head.weight is tied.tok_emb.weight
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert untied.num_parameters() - tied.num_parameters() == CFG.vocab_size * CFG.d_model


# ---------- initialization ----------


def test_weights_start_at_the_gpt2_scale_and_residual_writers_smaller():
    cfg = GPTConfig(vocab_size=64, context_length=16, d_model=256, n_layers=4, n_heads=4)
    m = GPT(cfg)
    qkv = m.blocks[0].attn.qkv.weight.std().item()  # type: ignore[union-attr]
    out = m.blocks[0].attn.out_proj.weight.std().item()  # type: ignore[union-attr]
    down = m.blocks[0].ff.down.weight.std().item()  # type: ignore[union-attr]
    assert qkv == pytest.approx(0.02, rel=0.05)
    assert out == pytest.approx(0.02 / math.sqrt(8), rel=0.05)
    assert down == pytest.approx(0.02 / math.sqrt(8), rel=0.05)


def test_with_bias_the_biases_start_at_zero():
    m = GPT(GPTConfig(**{**CFG.to_dict(), "bias": True}))
    biases = [p for n, p in m.named_parameters() if n.endswith(".bias") and "norm" not in n]
    assert biases and all(torch.all(b == 0) for b in biases)


# ---------- what the architecture must guarantee ----------


@pytest.mark.parametrize("cut", [0, 5, 10])
def test_changing_later_tokens_never_changes_earlier_logits(cut: int):
    m, x = model(), ids()
    changed = x.clone()
    changed[:, cut + 1 :] = (changed[:, cut + 1 :] + 7) % 50
    a, _ = m(x)
    b, _ = m(changed)
    torch.testing.assert_close(a[:, : cut + 1], b[:, : cut + 1], atol=1e-5, rtol=0)


def test_position_embeddings_are_what_tells_a_text_from_its_reordering():
    # One layer, so the claim is exact: the last token attends to the token embeddings of the
    # prefix, and without positions only the SET of those matters. (With two or more layers the
    # causal mask alone lets earlier tokens carry some order, because each sees a different prefix.)
    m = model(GPTConfig(**{**CFG.to_dict(), "n_layers": 1}))
    x = ids(batch=1)
    swapped = x.clone()
    swapped[0, [2, 5]] = x[0, [5, 2]]  # reorder two tokens of the prefix, keep the last
    assert not torch.allclose(m(x)[0][:, -1], m(swapped)[0][:, -1], atol=1e-4)
    with torch.no_grad():
        m.pos_emb.weight.zero_()
    torch.testing.assert_close(m(x)[0][:, -1], m(swapped)[0][:, -1], atol=1e-5, rtol=1e-5)


def test_a_block_with_silent_sublayers_is_the_identity():
    block = Block(CFG).eval()
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.ff.down.weight.zero_()
    x = torch.randn(2, 5, CFG.d_model)
    assert torch.equal(block(x), x)  # the residual path passes the input through unchanged


def test_the_fast_attention_path_gives_the_same_model():
    slow = model()
    fast = GPT(GPTConfig(**{**CFG.to_dict(), "fast_attention": True})).eval()
    fast.load_state_dict(slow.state_dict())
    torch.testing.assert_close(fast(ids())[0], slow(ids())[0], atol=1e-5, rtol=1e-5)


def test_a_checkpoint_restores_the_same_predictions():
    a = model(seed=0)
    b = model(seed=5)
    assert not torch.equal(a(ids())[0], b(ids())[0])
    b.load_state_dict(a.state_dict())
    assert torch.equal(a(ids())[0], b(ids())[0])


def test_dropout_acts_in_training_only():
    m = GPT(GPTConfig(**{**CFG.to_dict(), "dropout": 0.3})).eval()
    assert torch.equal(m(ids())[0], m(ids())[0])
    m.train()
    assert not torch.equal(m(ids())[0], m(ids())[0])


def test_a_text_longer_than_the_context_is_refused():
    with pytest.raises(ValueError, match="context"):
        model()(ids(length=17))


# ---------- it can learn ----------


def test_it_memorizes_a_small_batch():
    cfg = GPTConfig(vocab_size=32, context_length=16, d_model=64, n_layers=2, n_heads=4)
    m = model(cfg).train()
    x = torch.randint(0, 32, (4, 16), generator=torch.Generator().manual_seed(3))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(200):
        _, loss = m(x, y)
        assert loss is not None
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first is not None and first > 3.0 and loss.item() < 0.3


def test_the_output_map_reads_the_normalized_final_state():
    m, calls = model(), []
    m.final_norm.register_forward_hook(lambda mod, inp, out: calls.append(out))
    logits, _ = m(ids())
    assert len(calls) == 1
    torch.testing.assert_close(logits, m.lm_head(calls[0]))
