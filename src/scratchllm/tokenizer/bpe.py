"""Training a byte-level BPE vocabulary (Sennrich et al. 2016; bytes as in GPT-2).

The idea in one paragraph. Start with 256 tokens, one per byte, so any text can be written.
Count how often each two neighbouring tokens occur together inside the same pre-token (see
`pretokenize`). Take the most frequent pair, give it a new token id, and rewrite the corpus with
that pair joined. Repeat until the vocabulary has the wanted size. The ordered list of joined
pairs (the "merges") is the whole result: token 256 is merges[0], token 257 is merges[1], and so on.

Two choices that matter and that the code makes explicit:

* Counting is done on UNIQUE pre-tokens with their frequency, not on the running text. "the"
  occurring a million times is one entry with weight 1,000,000.
* After a merge only the words that contained the pair change, so the pair counts are updated
  for those words alone, and the best pair is found with a heap instead of a scan. This is the
  difference between minutes and days at 16k merges. `reference_merges` in the tests is the slow,
  obviously-correct version and must give the same merges.

Ties between pairs with the same count go to the pair with the smaller ids. That is arbitrary but
fixed, so the same corpus always gives the same vocabulary, whatever order Python iterates in.
"""

from __future__ import annotations

import heapq
import re
from collections import Counter
from collections.abc import Iterable, Mapping

from scratchllm.tokenizer.pretokenize import pretokenize

Pair = tuple[int, int]
BASE_VOCAB = 256  # ids 0..255 are the bytes themselves


def count_words(texts: Iterable[str], special_tokens: Iterable[str] = ()) -> Counter[bytes]:
    """Count the pre-tokens of all `texts`, as UTF-8 bytes.

    Text is cut at every special token first and the special tokens themselves are not counted,
    so a placeholder such as "<email>" never takes part in a merge: it will be one reserved id.
    """
    specials = sorted(special_tokens, key=len, reverse=True)  # longest first if one is a prefix
    cutter = re.compile("|".join(re.escape(s) for s in specials)) if specials else None
    counts: Counter[bytes] = Counter()
    for text in texts:
        for chunk in cutter.split(text) if cutter else [text]:
            counts.update(piece.encode("utf-8") for piece in pretokenize(chunk))
    return counts


def merge_word(word: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace each non-overlapping occurrence of `pair` in `word`, scanning left to right."""
    out: list[int] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train_merges(word_counts: Mapping[bytes, int], num_merges: int) -> list[Pair]:
    """Learn up to `num_merges` merges from pre-token counts. Stops early if nothing is left."""
    words = [list(word) for word in word_counts]
    freqs = list(word_counts.values())

    pair_counts: dict[Pair, int] = {}
    where: dict[Pair, set[int]] = {}  # pair -> indices of words that may contain it
    for index, word in enumerate(words):
        for pair in zip(word, word[1:], strict=False):
            pair_counts[pair] = pair_counts.get(pair, 0) + freqs[index]
            where.setdefault(pair, set()).add(index)

    # Max-heap by count (negated), then smallest pair. Entries go stale when a count changes; an
    # entry is valid only if it still matches pair_counts, so stale ones are skipped when popped.
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    while len(merges) < num_merges:
        best = None
        while heap:
            negative_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair) == -negative_count:
                best = pair
                break
        if best is None:
            break
        new_id = BASE_VOCAB + len(merges)
        merges.append(best)

        changed: set[Pair] = set()
        for index in where.pop(best):
            word = words[index]
            merged = merge_word(word, best, new_id)
            if len(merged) == len(word):
                continue  # the index was stale: an earlier merge already removed the pair here
            freq = freqs[index]
            for pair in zip(word, word[1:], strict=False):
                pair_counts[pair] -= freq
                changed.add(pair)
            for pair in zip(merged, merged[1:], strict=False):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                where.setdefault(pair, set()).add(index)
                changed.add(pair)
            words[index] = merged
        for pair in changed:
            count = pair_counts[pair]
            if count > 0:
                heapq.heappush(heap, (-count, pair))
            else:
                del pair_counts[pair]
    return merges


def train_bpe(
    texts: Iterable[str], vocab_size: int, special_tokens: Iterable[str] = ()
) -> list[Pair]:
    """Merges for a vocabulary of `vocab_size` ids: 256 bytes + merges + the special tokens."""
    specials = list(special_tokens)
    num_merges = vocab_size - BASE_VOCAB - len(specials)
    if num_merges < 0:
        raise ValueError(
            f"vocab_size {vocab_size} cannot hold {BASE_VOCAB} bytes and {len(specials)} specials"
        )
    return train_merges(count_words(texts, specials), num_merges)


def vocab_from_merges(merges: list[Pair]) -> list[bytes]:
    """The bytes each id stands for: ids 0..255 are single bytes, then one entry per merge."""
    vocab = [bytes([i]) for i in range(BASE_VOCAB)]
    for first, second in merges:
        vocab.append(vocab[first] + vocab[second])
    return vocab
