"""Tokenizer fertility: how many tokens a tokenizer needs for the same text.

Fertility (Rust et al. 2021) is tokens per word. It is a ratio of two counts and both can be
argued with, so this module reports four numbers and the report prints all of them:

* tokens per word: the word is whatever `str.split()` finds. For Arabic that is a rough unit,
  because a written word often carries a conjunction, a preposition or a pronoun glued to it.
  The number is comparable between tokenizers on the same text, not between languages.
* characters per token and bytes per token: independent of the word definition. For one text,
  a larger value means fewer tokens, so a shorter sequence for the same model context.
* tokens per 1,000 characters: the inverse view of the same thing, easy to read as a cost.

All of them are measured on the same normalized text for every tokenizer being compared, so the
only thing that differs is the tokenizer. Special tokens of our own tokenizer (`<email>`,
`<phone>`) count as one token; an outside tokenizer sees them as ordinary text, which favours
ours slightly. `special` counts how many there were, so the size of that effect is visible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass
class Counts:
    docs: int = 0
    chars: int = 0
    bytes: int = 0
    words: int = 0
    tokens: int = 0
    special: int = 0  # special-token ids produced (only our tokenizer has any)

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / self.words

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.tokens

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / self.tokens

    @property
    def tokens_per_1000_chars(self) -> float:
        return 1000 * self.tokens / self.chars


def measure(
    encode: Callable[[str], list[int]],
    texts: Iterable[str],
    is_special: Callable[[int], bool] | None = None,
) -> Counts:
    counts = Counts()
    for text in texts:
        ids = encode(text)
        counts.docs += 1
        counts.chars += len(text)
        counts.bytes += len(text.encode("utf-8"))
        counts.words += len(text.split())
        counts.tokens += len(ids)
        if is_special is not None:
            counts.special += sum(is_special(i) for i in ids)
    return counts
