import random
from collections import Counter

import pytest

from scratchllm.tokenizer.bpe import (
    BASE_VOCAB,
    Pair,
    count_words,
    merge_word,
    train_bpe,
    train_merges,
    vocab_from_merges,
)


def reference_merges(word_counts: dict[bytes, int], num_merges: int) -> list[Pair]:
    """The slow and obviously correct trainer: recount every pair from scratch at every step."""
    words = {tuple(word): count for word, count in word_counts.items()}
    merges: list[Pair] = []
    for _ in range(num_merges):
        pairs: Counter[Pair] = Counter()
        for word, count in words.items():
            for pair in zip(word, word[1:], strict=False):
                pairs[pair] += count
        if not pairs:
            break
        best = min(pairs, key=lambda pair: (-pairs[pair], pair))
        new_id = BASE_VOCAB + len(merges)
        merges.append(best)
        merged: Counter[tuple[int, ...]] = Counter()
        for word, count in words.items():
            merged[tuple(merge_word(list(word), best, new_id))] += count
        words = dict(merged)
    return merges


# ---------- merge_word ----------


@pytest.mark.parametrize(
    ("word", "pair", "expected"),
    [
        ([1, 2, 3], (1, 2), [9, 3]),
        ([1, 2, 1, 2], (1, 2), [9, 9]),
        ([1, 1, 1], (1, 1), [9, 1]),  # left to right, no overlap: "aaa" -> "aa" + "a"
        ([1, 1, 1, 1], (1, 1), [9, 9]),
        ([3, 4], (1, 2), [3, 4]),
        ([], (1, 2), []),
    ],
)
def test_merge_word(word: list[int], pair: Pair, expected: list[int]):
    assert merge_word(word, pair, 9) == expected


# ---------- counting ----------


def test_count_words_counts_pretokens_as_bytes():
    counts = count_words(["a b a", "a"])
    assert counts == {b"a": 2, b" b": 1, b" a": 1}


def test_special_tokens_cut_the_text_and_are_never_counted():
    counts = count_words(["hi<email>there <phone>x"], special_tokens=["<email>", "<phone>"])
    assert counts == {b"hi": 1, b"there": 1, b" ": 1, b"x": 1}


def test_count_words_prefers_the_longest_special_token():
    counts = count_words(["a<s>b<s2>c"], special_tokens=["<s>", "<s2>"])
    assert counts == {b"a": 1, b"b": 1, b"c": 1}


# ---------- the algorithm ----------


def test_the_example_of_the_paper_without_end_of_word_marker():
    # Sennrich et al.'s toy corpus: low x5, lower x2, newest x6, widest x3. (e,s) and (s,t) tie at
    # 9 and the smaller ids win, so "es" comes first; then "est"; then (l,o) and (o,w) tie at 7.
    counts = {b"low": 5, b"lower": 2, b"newest": 6, b"widest": 3}
    merges = train_merges(counts, 5)
    vocab = vocab_from_merges(merges)
    assert [vocab[BASE_VOCAB + i] for i in range(5)] == [b"es", b"est", b"lo", b"low", b"ew"]


def test_ids_of_merges_refer_to_earlier_tokens_only():
    counts = count_words(["the cat sat on the mat. the cat sat."])
    merges = train_merges(counts, 30)
    for i, (first, second) in enumerate(merges):
        assert first < BASE_VOCAB + i and second < BASE_VOCAB + i


def test_stops_early_when_nothing_is_left_to_merge():
    # every word is a single byte: there is no pair at all
    assert train_merges({b"a": 4, b"b": 2}, 10) == []
    # "ab" x3 can only be merged once
    assert train_merges({b"ab": 3}, 10) == [(97, 98)]


def test_the_same_corpus_always_gives_the_same_merges():
    texts = ["abracadabra alakazam abracadabra", "banana bandana"]
    first = train_merges(count_words(texts), 20)
    second = train_merges(count_words(reversed(texts)), 20)
    assert first == second


@pytest.mark.parametrize("seed", range(8))
def test_the_fast_trainer_gives_the_same_merges_as_the_reference(seed: int):
    # a tiny alphabet makes counts collide constantly, which is what stresses the tie-breaking
    rng = random.Random(seed)
    alphabet = "abc" if seed % 2 else "abcdef"
    words = {
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 9))).encode(): rng.randrange(
            1, 6
        )
        for _ in range(60)
    }
    assert train_merges(words, 40) == reference_merges(words, 40)


def test_the_fast_trainer_matches_the_reference_on_arabic_bytes():
    texts = ["ذهب الولد إلى المدرسة", "ذهبت البنت إلى المدرسة", "مُحَمَّد ذهب إلى البيت"] * 3
    counts = count_words(texts)
    assert train_merges(counts, 60) == reference_merges(dict(counts), 60)


def test_a_repeated_arabic_word_becomes_one_token():
    word = " مرحبا".encode()
    merges = train_merges(count_words([" مرحبا"] * 20), len(word) - 1)
    assert word in vocab_from_merges(merges)


# ---------- the vocabulary ----------


def test_vocab_from_merges():
    vocab = vocab_from_merges([(97, 98), (256, 99)])
    assert len(vocab) == BASE_VOCAB + 2
    assert vocab[0] == b"\x00" and vocab[255] == b"\xff"
    assert vocab[256] == b"ab" and vocab[257] == b"abc"


def test_vocab_size_counts_the_bytes_and_the_special_tokens():
    texts = ["abcabcabc abcabc"] * 3
    merges = train_bpe(texts, vocab_size=BASE_VOCAB + 5 + 2, special_tokens=["<a>", "<b>"])
    assert len(merges) == 5


def test_a_vocab_too_small_for_the_bytes_is_refused():
    with pytest.raises(ValueError):
        train_bpe(["x"], vocab_size=100)
    with pytest.raises(ValueError):
        train_bpe(["x"], vocab_size=BASE_VOCAB + 1, special_tokens=["<a>", "<b>"])
