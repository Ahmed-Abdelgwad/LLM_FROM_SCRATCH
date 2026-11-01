"""The trained tokenizer: text -> token ids -> text.

A `Tokenizer` is nothing but the ordered list of merges that `train_bpe` learned, plus the special
tokens. Ids are laid out like this:

    0..255                          the 256 bytes
    256..256+len(merges)-1          one id per merge, in the order they were learned
    then                            the special tokens, in the order given

Encoding a piece of text replays the training. Start from the piece's bytes and repeatedly apply
the merge that was learned EARLIEST among the pairs present, until no pair that is in the merge
list is left. "Earliest learned" and not "most frequent in this word" is the point: the merges
were ranked by frequency over the whole corpus, and replaying them in that order is what makes
encoding give the same pieces training saw. `tests/test_tokenizer.py` checks this against the
literal definition (apply merge 0 to the word, then merge 1, ...).

Not done here, on purpose: normalization. `encode` takes the text it is given. Whoever prepares
training text and whoever prepares a prompt must both call `normalize` first (D-013), or the
model is trained on one spelling and asked about another.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from scratchllm.tokenizer.bpe import BASE_VOCAB, Pair, merge_word, train_bpe, vocab_from_merges
from scratchllm.tokenizer.pretokenize import PATTERN, pretokenize

FORMAT = "scratchllm-bpe"
VERSION = 1
_CACHE_LIMIT = 200_000  # pieces remembered; words repeat a lot, but memory is small here


class Tokenizer:
    def __init__(self, merges: Iterable[Pair], special_tokens: Iterable[str] = ()) -> None:
        self.merges: list[Pair] = [(int(a), int(b)) for a, b in merges]
        self.special_tokens: list[str] = list(special_tokens)
        self._check()

        self._ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(self.merges)}
        self._vocab: list[bytes] = vocab_from_merges(self.merges)
        first_special = len(self._vocab)
        self._special_ids = {tok: first_special + i for i, tok in enumerate(self.special_tokens)}
        self._special_by_id = {i: tok for tok, i in self._special_ids.items()}
        # longest first, so a special token that is a prefix of another does not hide it
        longest_first = sorted(self.special_tokens, key=len, reverse=True)
        self._cutter = (
            re.compile("(" + "|".join(re.escape(tok) for tok in longest_first) + ")")
            if longest_first
            else None
        )
        self._cache: dict[bytes, list[int]] = {}

    def _check(self) -> None:
        for rank, (first, second) in enumerate(self.merges):
            if not (0 <= first < BASE_VOCAB + rank and 0 <= second < BASE_VOCAB + rank):
                raise ValueError(f"merge {rank} {(first, second)} uses an id not created yet")
        if len(set(self.merges)) != len(self.merges):
            raise ValueError("the same pair is merged twice")
        if any(not tok for tok in self.special_tokens):
            raise ValueError("a special token cannot be empty")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("duplicate special token")

    @classmethod
    def train(
        cls, texts: Iterable[str], vocab_size: int, special_tokens: Iterable[str] = ()
    ) -> Tokenizer:
        specials = list(special_tokens)
        return cls(train_bpe(texts, vocab_size, specials), specials)

    def truncated(self, vocab_size: int) -> Tokenizer:
        """The tokenizer a training run with a smaller `vocab_size` would have produced.

        BPE learns merges one at a time, so the first k merges of a longer training are exactly
        the merges of a training that stopped at k (`tests/test_tokenizer.py` checks it). One long
        training therefore gives every smaller vocabulary, which is how the sizes are compared.
        """
        keep = vocab_size - BASE_VOCAB - len(self.special_tokens)
        if not 0 <= keep <= len(self.merges):
            raise ValueError(
                f"vocab_size must be between {self.vocab_size - len(self.merges)} "
                f"and {self.vocab_size}, got {vocab_size}"
            )
        return Tokenizer(self.merges[:keep], self.special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + len(self.special_tokens)

    def special_id(self, token: str) -> int:
        return self._special_ids[token]

    # ------------------------------------------------------------------ encode

    def _encode_piece(self, piece: bytes) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece)
        while len(ids) > 1:
            pairs = zip(ids, ids[1:], strict=False)
            best = min(pairs, key=lambda pair: self._ranks.get(pair, sys.maxsize))
            rank = self._ranks.get(best)
            if rank is None:
                break  # no pair left that was ever merged
            ids = merge_word(ids, best, BASE_VOCAB + rank)
        if len(self._cache) < _CACHE_LIMIT:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, parse_special: bool = True) -> list[int]:
        """Token ids of `text`.

        With `parse_special` the special-token strings found in the text become their reserved
        ids (that is how "<email>" in cleaned text reaches the model as one token). Without it
        they are ordinary characters, which is what to use on untrusted user input.
        """
        if self._cutter is None or not parse_special:
            return self._encode_text(text)
        ids: list[int] = []
        for index, chunk in enumerate(self._cutter.split(text)):  # normal, special, normal, ...
            if index % 2 == 1:
                ids.append(self._special_ids[chunk])
            else:
                ids.extend(self._encode_text(chunk))
        return ids

    def _encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in pretokenize(text):
            ids.extend(self._encode_piece(piece.encode("utf-8")))
        return ids

    # ------------------------------------------------------------------ decode

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """Text of `ids`.

        A model can emit ids whose bytes stop in the middle of a character (an Arabic letter is
        two bytes). `errors="replace"` shows such a fragment as U+FFFD; "strict" raises instead.
        """
        out = bytearray()
        for token_id in ids:
            if 0 <= token_id < len(self._vocab):
                out += self._vocab[token_id]
            elif token_id in self._special_by_id:
                out += self._special_by_id[token_id].encode("utf-8")
            else:
                raise ValueError(f"unknown token id {token_id}")
        return out.decode("utf-8", errors=errors)

    # ------------------------------------------------------------------ files

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "format": FORMAT,
            "version": VERSION,
            "pattern": PATTERN.pattern,  # the pre-tokenizer is part of the tokenizer
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": self.special_tokens,
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT or payload.get("version") != VERSION:
            raise ValueError(f"{path} is not a {FORMAT} v{VERSION} file")
        if payload["pattern"] != PATTERN.pattern:
            raise ValueError(
                "this file was trained with a different pre-tokenizer pattern; "
                "encoding with the current one would not match the training"
            )
        return cls([(a, b) for a, b in payload["merges"]], payload["special_tokens"])
