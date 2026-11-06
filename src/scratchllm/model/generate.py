"""Autoregressive text generation: the model writes one token at a time.

Each step feeds the text so far to the model, takes the scores it gives for the NEXT token (the
last position's row of logits), picks one token from them, appends it, and repeats. The ways to
pick are the only choices in this file:

* greedy (temperature 0): always the highest-scoring token. Deterministic, but it tends to loop
  ("the the the ...") because the most likely continuation of a repetition is more repetition.
* sampling: draw from softmax(logits / temperature). Temperature below 1 sharpens the
  distribution toward the best tokens, above 1 flattens it toward uniform.
* top-k: before sampling, keep only the k best-scoring tokens and renormalize, so the rare,
  low-quality tail can never be drawn. (Ties with the k-th score are kept.)

Stopping: when `eos_id` is generated the sequence is finished. Later steps pad it with `eos_id`,
and generation ends early once every sequence in the batch is finished. The returned ids include
the first `eos_id`; callers cut the text there.

The model only sees its last `context_length` tokens (its position embeddings stop there), so
when the text grows past that, the oldest tokens are dropped from the model's view while still
being kept in the output.

Key/value cache (D-030). A model that offers `new_cache()` is run with one: the prompt goes in
once and each later step feeds only the newest token, so a step costs one position instead of the
whole window. Once the text outgrows the window the cache cannot help (the model's view slides,
so every position's numbers change) and the remaining steps recompute the window as before.
`use_cache=False` turns it off.

Equivalence with the uncached computation is numerical, not bitwise, and the text is not
guaranteed to be the same:

* The logits agree to rounding: about 1e-5 in float32 (tested on four architecture variants).
* The text was identical in every float32 case tried (CPU tests; six prompts on two real
  checkpoints, greedy and sampled). That is an observation, not a guarantee: a near-tie between two
  tokens can be decided differently by a different order of floating-point operations.
* Under bfloat16 autocast, which the GPU commands use, the text differs on some prompts (D-030):
  the cached and the uncached logits each stray from the float32 ones by about 0.2, so an
  occasional token choice, and everything after it, can differ. Neither text is "the" right one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from scratchllm.data.tokens import END_OF_TEXT
from scratchllm.tokenizer.normalize import normalize
from scratchllm.tokenizer.tokenizer import Tokenizer


class _LanguageModel(Protocol):
    """What `generate` needs of a model: `cfg.context_length`, a mode flag, and a call that maps
    ids (batch, T) to (logits (batch, T, vocab), loss or None). `GPT` fits; so do test doubles."""

    cfg: Any
    training: bool
    __call__: Callable[..., Any]

    def eval(self) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...


def _pick(
    logits: torch.Tensor, temperature: float, top_k: int | None, generator: torch.Generator | None
) -> torch.Tensor:
    """One token per row of `logits` (batch, vocab)."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: _LanguageModel,
    ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Continue `ids` (batch, T0) or (T0,) by up to `max_new_tokens` tokens.

    Returns (batch, T0 + n) with n <= max_new_tokens (fewer only if every sequence hit `eos_id`).
    `temperature` 0 means greedy. `generator` makes sampling reproducible; it must live on the
    same device as the model.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")
    single = ids.dim() == 1
    out = ids.unsqueeze(0) if single else ids
    if out.size(1) == 0:
        raise ValueError("the prompt needs at least one token")
    context = model.cfg.context_length
    cache = model.new_cache() if use_cache and hasattr(model, "new_cache") else None
    finished = torch.zeros(out.size(0), dtype=torch.bool, device=out.device)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            if cache is not None and cache.length == 0:  # first step: the prompt, once
                logits, _ = model(out[:, -context:], cache=cache)
            elif cache is not None and cache.length < context and cache.length == out.size(1) - 1:
                logits, _ = model(out[:, -1:], cache=cache)  # only the newest token
            else:  # no cache, or the text has outgrown the window: recompute it
                logits, _ = model(out[:, -context:])
            token = _pick(logits[:, -1, :].float(), temperature, top_k, generator)
            if eos_id is not None:
                token = torch.where(finished, torch.full_like(token, eos_id), token)
                finished = finished | (token == eos_id)
            out = torch.cat([out, token[:, None]], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
    finally:
        model.train(was_training)
    return out[0] if single else out


@dataclass(frozen=True)
class Sample:
    prompt: str
    completion: str
    new_tokens: int
    stopped_by_eos: bool


def generate_text(
    model: _LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    use_cache: bool = True,
) -> Sample:
    """Text in, text out.

    The prompt goes through the same `normalize` as the training text (D-013, D-015) and is
    tokenized with special-token parsing off, so "<email>" typed by a user is plain text. It is
    preceded by the end-of-text id, because that is what precedes the first word of every document
    in the training stream; an empty prompt therefore means "start a new document".
    """
    eot = tokenizer.special_id(END_OF_TEXT)
    start = [eot, *tokenizer.encode(normalize(prompt), parse_special=False)]
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
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
    return Sample(prompt, tokenizer.decode(new), len(new), stopped)
