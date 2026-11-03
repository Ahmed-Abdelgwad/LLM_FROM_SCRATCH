from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from scratchllm.data.tokens import END_OF_TEXT
from scratchllm.model.generate import generate, generate_text
from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.tokenizer.tokenizer import Tokenizer

V = 8


class FakeLM:
    """A language model whose next-token scores are given by `rule(ids) -> (batch, V)`; it records
    how many tokens it was shown each time."""

    def __init__(self, rule: Callable[[torch.Tensor], torch.Tensor], context: int = 6) -> None:
        self.rule, self.cfg = rule, SimpleNamespace(context_length=context)
        self.training, self.seen = True, []

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, ids: torch.Tensor):
        assert not torch.is_grad_enabled()
        self.seen.append(ids.size(1))
        last = self.rule(ids)
        logits = torch.zeros(ids.size(0), ids.size(1), V)
        logits[:, -1, :] = last
        return logits, None


def favour(token: int, strength: float = 10.0) -> torch.Tensor:
    row = torch.zeros(V)
    row[token] = strength
    return row


def counting(ids: torch.Tensor) -> torch.Tensor:
    """Always prefers (last token + 1) mod V."""
    return torch.stack([favour(int(t) + 1 if int(t) + 1 < V else 0) for t in ids[:, -1]])


# ---------- greedy ----------


def test_greedy_follows_the_highest_score_and_keeps_the_prompt():
    out = generate(FakeLM(counting), torch.tensor([[1, 2]]), 4, temperature=0)
    assert out.tolist() == [[1, 2, 3, 4, 5, 6]]


def test_greedy_is_deterministic_and_equals_top_k_of_one():
    torch.manual_seed(0)
    model = FakeLM(
        lambda ids: torch.randn(
            ids.size(0), V, generator=torch.Generator().manual_seed(int(ids.sum()))
        )
    )
    prompt = torch.tensor([[3, 1, 4]])
    a = generate(model, prompt, 10, temperature=0)
    b = generate(model, prompt, 10, temperature=0)
    c = generate(
        model, prompt, 10, temperature=1.0, top_k=1, generator=torch.Generator().manual_seed(1)
    )
    assert torch.equal(a, b) and torch.equal(a, c)


def test_a_one_dimensional_prompt_gives_a_one_dimensional_answer():
    out = generate(FakeLM(counting), torch.tensor([1, 2]), 2, temperature=0)
    assert out.tolist() == [1, 2, 3, 4]


def test_zero_new_tokens_returns_the_prompt():
    assert generate(FakeLM(counting), torch.tensor([[1, 2]]), 0).tolist() == [[1, 2]]


# ---------- end of sequence ----------


def test_generation_stops_at_the_end_token_and_includes_it():
    model = FakeLM(counting)
    out = generate(model, torch.tensor([[1]]), 20, temperature=0, eos_id=5)
    assert out.tolist() == [[1, 2, 3, 4, 5]]
    assert len(model.seen) == 4  # it did not keep computing


def test_in_a_batch_finished_rows_are_padded_and_the_run_ends_when_all_are_done():
    out = generate(FakeLM(counting), torch.tensor([[1], [3]]), 20, temperature=0, eos_id=5)
    assert out.tolist() == [[1, 2, 3, 4, 5], [3, 4, 5, 5, 5]]


def test_without_an_end_token_it_runs_to_the_limit():
    assert generate(FakeLM(counting), torch.tensor([[1]]), 9, temperature=0).size(1) == 10


# ---------- context length ----------


def test_the_model_never_sees_more_than_its_context_but_the_output_keeps_everything():
    model = FakeLM(counting, context=6)
    out = generate(
        model, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1]]), 5, temperature=0
    )  # prompt > context
    assert max(model.seen) == 6
    assert out.size(1) == 15
    assert out[0, -5:].tolist() == [2, 3, 4, 5, 6]
    assert out[0, :10].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]


def test_generating_far_past_the_context_works():
    model = FakeLM(counting, context=4)
    out = generate(model, torch.tensor([[0]]), 30, temperature=0)
    assert out.size(1) == 31 and max(model.seen) == 4


# ---------- temperature and top-k ----------


def frequencies(model, n: int = 6000, **kwargs) -> torch.Tensor:
    out = generate(
        model,
        torch.zeros(n, 1, dtype=torch.long),
        1,
        generator=torch.Generator().manual_seed(0),
        **kwargs,
    )
    return torch.bincount(out[:, 1], minlength=V).float() / n


GRADED = FakeLM(lambda ids: torch.arange(V, dtype=torch.float).expand(ids.size(0), V))


def test_sampling_follows_softmax_of_the_scores_at_temperature_one():
    expected = torch.softmax(torch.arange(V, dtype=torch.float), 0)
    torch.testing.assert_close(frequencies(GRADED, temperature=1.0), expected, atol=0.02, rtol=0)


def test_a_lower_temperature_concentrates_and_a_higher_one_spreads():
    cold, warm, hot = (frequencies(GRADED, temperature=t) for t in (0.3, 1.0, 3.0))
    assert cold[-1] > warm[-1] > hot[-1]
    torch.testing.assert_close(
        cold, torch.softmax(torch.arange(V, dtype=torch.float) / 0.3, 0), atol=0.02, rtol=0
    )
    torch.testing.assert_close(
        hot, torch.softmax(torch.arange(V, dtype=torch.float) / 3.0, 0), atol=0.02, rtol=0
    )
    assert hot[0] > 0.02 and hot[-1] < 0.35  # flat compared with temperature 1


def test_top_k_never_draws_outside_the_k_best_and_renormalizes_inside():
    f = frequencies(GRADED, temperature=1.0, top_k=3)
    assert torch.all(f[:5] == 0)
    torch.testing.assert_close(
        f[5:], torch.softmax(torch.tensor([5.0, 6.0, 7.0]), 0), atol=0.02, rtol=0
    )


def test_top_k_larger_than_the_vocabulary_changes_nothing():
    a = frequencies(GRADED, temperature=1.0, top_k=1000)
    b = frequencies(GRADED, temperature=1.0)
    assert torch.equal(a, b)


def test_the_same_seed_gives_the_same_text_and_another_seed_a_different_one():
    model = FakeLM(lambda ids: torch.zeros(ids.size(0), V))  # all tokens equally likely
    prompt = torch.tensor([[0]])

    def run(seed: int) -> torch.Tensor:
        return generate(model, prompt, 30, generator=torch.Generator().manual_seed(seed))

    assert torch.equal(run(1), run(1)) and not torch.equal(run(1), run(2))


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": -1.0}, {"top_k": 0}, {"top_k": -2}, {"max_new_tokens": -1}],
)
def test_bad_arguments_are_refused(kwargs):
    args = {"max_new_tokens": 3, **kwargs}
    with pytest.raises(ValueError):
        generate(FakeLM(counting), torch.tensor([[1]]), **args)


def test_an_empty_prompt_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        generate(FakeLM(counting), torch.zeros(1, 0, dtype=torch.long), 3)


def test_the_models_training_mode_is_restored():
    model = FakeLM(counting)
    generate(model, torch.tensor([[1]]), 2, temperature=0)
    assert model.training is True
    model.eval()
    generate(model, torch.tensor([[1]]), 2, temperature=0)
    assert model.training is False


# ---------- with the real model and a real tokenizer ----------


CORPUS = ["ذهب الولد إلى المدرسة صباحا", "the boy went to school in the morning"] * 10


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.train(
        CORPUS, vocab_size=256 + 40 + 3, special_tokens=[END_OF_TEXT, "<email>", "<phone>"]
    )


def tiny_gpt(vocab: int) -> GPT:
    torch.manual_seed(0)
    return GPT(
        GPTConfig(vocab_size=vocab, context_length=16, d_model=32, n_layers=1, n_heads=4)
    ).eval()


def test_a_real_model_produces_valid_ids_and_keeps_the_prompt():
    model = tiny_gpt(50)
    prompt = torch.randint(0, 50, (3, 5))
    out = generate(
        model, prompt, 25, temperature=0.8, top_k=10, generator=torch.Generator().manual_seed(0)
    )
    assert out.shape == (3, 30) and torch.equal(out[:, :5], prompt)
    assert int(out.min()) >= 0 and int(out.max()) < 50


def test_generate_text_returns_the_text_before_the_end_of_text_id(tokenizer: Tokenizer):
    eot = tokenizer.special_id(END_OF_TEXT)
    word = tokenizer.encode(" school", parse_special=False)
    script = word + [eot]  # the model "writes" ' school' and then ends the document
    prompt_len = 1 + len(tokenizer.encode("the boy", parse_special=False))

    def rule(ids: torch.Tensor) -> torch.Tensor:
        row = torch.zeros(ids.size(0), tokenizer.vocab_size)
        row[:, script[min(ids.size(1) - prompt_len, len(script) - 1)]] = 10.0
        return row

    model = FakeLM(rule, context=64)
    model.__class__ = type("Wide", (FakeLM,), {"__call__": _wide_call(tokenizer.vocab_size)})
    sample = generate_text(model, tokenizer, "the boy", 20, temperature=0)
    assert sample.completion == " school" and sample.stopped_by_eos
    assert sample.new_tokens == len(word) and sample.prompt == "the boy"
    capped = generate_text(model, tokenizer, "the boy", 1, temperature=0)  # limit before the end
    assert not capped.stopped_by_eos and capped.new_tokens == 1


def test_the_real_model_can_be_sampled_as_text_with_a_seed(tokenizer: Tokenizer):
    model = tiny_gpt(tokenizer.vocab_size)
    a = generate_text(model, tokenizer, "the boy", 12, temperature=1.0, top_k=20, seed=3)
    b = generate_text(model, tokenizer, "the boy", 12, temperature=1.0, top_k=20, seed=3)
    c = generate_text(model, tokenizer, "the boy", 12, temperature=1.0, top_k=20, seed=4)
    assert a == b and a != c and a.new_tokens <= 12


def test_the_prompt_is_normalized_and_special_strings_are_plain_text(tokenizer: Tokenizer):
    shown: list[list[int]] = []
    width = tokenizer.vocab_size

    def rule(ids: torch.Tensor) -> torch.Tensor:
        shown.append(ids[0].tolist())
        return torch.zeros(ids.size(0), width)

    model = FakeLM(rule, context=64)
    model.__class__ = type("Wide", (FakeLM,), {"__call__": _wide_call(width)})
    generate_text(model, tokenizer, "ذهب  الولد\u00a0<email>", 1, temperature=0)
    first = shown[0]
    assert first[0] == tokenizer.special_id(END_OF_TEXT)  # starts like a document of the stream
    assert tokenizer.special_id("<email>") not in first  # typed text, not the special token
    assert first[1:] == tokenizer.encode("ذهب الولد <email>", parse_special=False)  # normalized


def _wide_call(width: int):
    def call(self, ids: torch.Tensor):
        self.seen.append(ids.size(1))
        last = self.rule(ids)
        return torch.zeros(ids.size(0), ids.size(1), width) + last[:, None, :], None

    return call


def test_an_empty_prompt_starts_a_new_document(tokenizer: Tokenizer):
    model = tiny_gpt(tokenizer.vocab_size)
    assert generate_text(model, tokenizer, "", 5, temperature=0).new_tokens <= 5
