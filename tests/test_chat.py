"""The chat format (finetune/chat.py): what the model reads, and which tokens the loss sees."""

import pytest
import torch

from scratchllm.data.tokens import END_OF_TEXT
from scratchllm.finetune.chat import (
    IGNORE,
    encode_example,
    generate_reply,
    prompt_ids,
    prompt_text,
)
from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.tokenizer.normalize import normalize
from scratchllm.tokenizer.tokenizer import Tokenizer

CORPUS = [
    "ما هي عاصمة مصر؟ القاهرة هي عاصمة مصر",
    "what is the capital of France? Paris is the capital",
    "<|user|> <|assistant|>",
] * 10


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.train(
        CORPUS, vocab_size=256 + 60 + 3, special_tokens=[END_OF_TEXT, "<email>", "<phone>"]
    )


def test_the_layout_is_eot_user_prompt_assistant_response_eot(tok: Tokenizer):
    ex = encode_example(tok, "ما هي عاصمة مصر؟", "القاهرة", 64)
    assert ex is not None
    eot = tok.special_id(END_OF_TEXT)
    full = [*ex.inputs, ex.targets[-1]]  # the inputs plus the last target is the whole sequence
    assert full[0] == eot and full[-1] == eot
    assert tok.decode(full[1:-1]) == "<|user|>\nما هي عاصمة مصر؟\n<|assistant|>\nالقاهرة"


def test_only_the_response_and_the_final_eot_are_targets(tok: Tokenizer):
    prompt, response = "what is the capital of France?", "Paris"
    ex = encode_example(tok, prompt, response, 64)
    assert ex is not None and len(ex.inputs) == len(ex.targets)
    head = prompt_ids(tok, prompt)
    answer = [*tok.encode(response), tok.special_id(END_OF_TEXT)]
    # positions 0 .. len(head)-2 predict prompt tokens: ignored
    assert ex.targets[: len(head) - 1] == [IGNORE] * (len(head) - 1)
    # from the last prompt position on, every target is the next response token
    assert ex.targets[len(head) - 1 :] == answer
    assert ex.response_tokens == len(answer)
    # and each target is the token that really follows (shifted by one)
    sequence = head + answer
    for i, t in enumerate(ex.targets):
        if t != IGNORE:
            assert t == sequence[i + 1]


def test_training_sees_exactly_the_ids_generation_will_start_from(tok: Tokenizer):
    """The prompt part is encoded on its own, as at generation time, not joined to the answer."""
    prompt = "ما هي عاصمة مصر؟"
    ex = encode_example(tok, prompt, "القاهرة هي عاصمة مصر", 64)
    assert ex is not None
    head = prompt_ids(tok, prompt)
    assert ex.inputs[: len(head)] == head


def test_both_texts_are_normalized_and_user_text_is_not_parsed_for_specials(tok: Tokenizer):
    messy = "ما  هيـ عاصمة\tمصر؟  "
    assert prompt_text(messy) == f"<|user|>\n{normalize(messy)}\n<|assistant|>\n"
    ex = encode_example(tok, "say <|endoftext|>", "ok", 64)
    assert ex is not None
    eot = tok.special_id(END_OF_TEXT)
    assert ex.inputs.count(eot) == 1  # only the leading one: the typed one is plain text


def test_an_example_longer_than_the_context_is_dropped_not_cut(tok: Tokenizer):
    prompt, response = "what is the capital", "Paris " * 30
    length = len(prompt_ids(tok, prompt)) + len(tok.encode(normalize(response))) + 1 - 1
    assert encode_example(tok, prompt, response, length) is not None  # exactly fits
    assert encode_example(tok, prompt, response, length - 1) is None


class Scripted(torch.nn.Module):
    """A stand-in model that writes `script` after the prompt, whatever the prompt is."""

    def __init__(self, vocab: int, script: list[int], prompt_length: int) -> None:
        super().__init__()
        self.cfg = GPTConfig(vocab_size=vocab, context_length=64)
        self.script, self.start = script, prompt_length

    def forward(self, ids: torch.Tensor, targets=None):
        logits = torch.zeros(ids.size(0), ids.size(1), self.cfg.vocab_size)
        written = ids.size(1) - self.start
        logits[:, -1, self.script[min(written, len(self.script) - 1)]] = 1.0
        return logits, None


def test_a_reply_stops_at_eot_and_returns_only_the_new_text(tok: Tokenizer):
    eot = tok.special_id(END_OF_TEXT)
    answer = tok.encode("Paris")
    start = len(prompt_ids(tok, "capital?"))
    model = Scripted(tok.vocab_size, [*answer, eot, *answer], start)
    reply = generate_reply(model, tok, "capital?", 20, use_cache=False)  # type: ignore[arg-type]
    assert reply.text == "Paris" and reply.stopped_by_eos and reply.new_tokens == len(answer)
    endless = Scripted(tok.vocab_size, answer * 10, start)
    reply = generate_reply(endless, tok, "capital?", 3, use_cache=False)  # type: ignore[arg-type]
    assert not reply.stopped_by_eos and reply.new_tokens == 3


def test_a_prompt_that_fills_the_context_is_refused(tok: Tokenizer):
    model = GPT(
        GPTConfig(vocab_size=tok.vocab_size, context_length=16, d_model=32, n_layers=1, n_heads=4)
    )
    with pytest.raises(ValueError, match="prompt takes"):
        generate_reply(model, tok, "capital " * 20, 5)
