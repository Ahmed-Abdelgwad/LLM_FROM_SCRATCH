"""The chat format: how a (prompt, response) pair becomes a training sequence, and how a reply is
asked for.

One example is laid out as

    <|endoftext|><|user|>\\n{prompt}\\n<|assistant|>\\n{response}<|endoftext|>

and the model is trained on the response only. The prompt and the markers are input the model
reads but is never asked to predict (their targets are `IGNORE`), so the loss measures one thing:
how well it answers. The final `<|endoftext|>` IS a target: that is how the model learns to stop.

Choices, and why:

* The markers are plain text, not new special tokens. New ids would need new rows in the
  embedding matrix, starting from nothing, and LoRA freezes that matrix (`lora.py`). The cost is
  a few tokens per marker; the markers do not occur in the pretraining text, so they still mark
  the turns unambiguously.
* The leading `<|endoftext|>` is there because it precedes every document in the pretraining
  stream (`data/tokens.py`), so the example starts where the model expects a new text to start;
  `generate_text` does the same.
* The prompt part and the response are encoded SEPARATELY. A reply is generated after
  `encode(prompt part)`, so training must see exactly those ids before the response: encoding the
  joined string could merge the last bytes of the prompt part with the first of the response into
  a token the model would never see at generation time.
* Both texts go through `normalize` (D-013), like every text the model has read or will read.
* An example longer than the context is dropped, not cut. Cutting the response and still asking
  for `<|endoftext|>` would teach the model to stop in mid-sentence; cutting the prompt would
  teach it to answer a question it cannot see. The caller counts what was dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scratchllm.data.tokens import END_OF_TEXT
from scratchllm.model.generate import generate
from scratchllm.model.gpt import GPT
from scratchllm.tokenizer.normalize import normalize
from scratchllm.tokenizer.tokenizer import Tokenizer

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
IGNORE = -100  # the target value `F.cross_entropy` skips (its default `ignore_index`)


def prompt_text(prompt: str) -> str:
    """The text the model reads before it answers."""
    return f"{USER}\n{normalize(prompt)}\n{ASSISTANT}\n"


def prompt_ids(tokenizer: Tokenizer, prompt: str) -> list[int]:
    """End-of-text, then the prompt part. Text typed by a user is never parsed for special
    tokens, so a literal "<|endoftext|>" in it is plain text."""
    eot = tokenizer.special_id(END_OF_TEXT)
    return [eot, *tokenizer.encode(prompt_text(prompt), parse_special=False)]


@dataclass(frozen=True)
class EncodedExample:
    """`inputs[i]` is read at position i and `targets[i]` is what position i must predict, or
    `IGNORE`. Both have the same length, at most the context length."""

    inputs: list[int]
    targets: list[int]

    @property
    def response_tokens(self) -> int:
        """Tokens the loss is computed on: the response and the final end-of-text."""
        return sum(t != IGNORE for t in self.targets)


def encode_example(
    tokenizer: Tokenizer, prompt: str, response: str, context_length: int
) -> EncodedExample | None:
    """The training sequence of one pair, or None if it does not fit `context_length`."""
    head = prompt_ids(tokenizer, prompt)
    answer = [
        *tokenizer.encode(normalize(response), parse_special=False),
        tokenizer.special_id(END_OF_TEXT),
    ]
    sequence = head + answer
    if len(sequence) - 1 > context_length:
        return None
    # position i predicts sequence[i + 1]; the first response token, sequence[len(head)], is
    # predicted from position len(head) - 1
    targets = [IGNORE] * (len(head) - 1) + answer
    return EncodedExample(inputs=sequence[:-1], targets=targets)


@dataclass(frozen=True)
class Reply:
    prompt: str
    text: str
    new_tokens: int
    stopped_by_eos: bool


def generate_reply(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    use_cache: bool = True,
) -> Reply:
    """The model's answer to `prompt`, in the format it was fine-tuned on. Greedy by default."""
    start = prompt_ids(tokenizer, prompt)
    if len(start) >= model.cfg.context_length:
        raise ValueError(f"the prompt takes {len(start)} tokens, the context holds fewer")
    eot = tokenizer.special_id(END_OF_TEXT)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    out = generate(
        model,
        torch.tensor(start, device=device),
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eot,
        generator=generator,
        use_cache=use_cache,
    )
    new = out[len(start) :].tolist()
    stopped = eot in new
    if stopped:
        new = new[: new.index(eot)]
    return Reply(prompt, tokenizer.decode(new), len(new), stopped)
