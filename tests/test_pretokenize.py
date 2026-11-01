import random

import pytest

from scratchllm.tokenizer.pretokenize import pretokenize


def test_english_pieces_carry_their_leading_space():
    assert pretokenize("Hello, world!") == ["Hello", ",", " world", "!"]


def test_contractions_are_split_off():
    assert pretokenize("I don't know") == ["I", " don", "'t", " know"]


def test_arabic_words_carry_their_leading_space():
    assert pretokenize("ذهب الولد") == ["ذهب", " الولد"]


def test_a_diacritized_arabic_word_is_one_piece():
    # GPT-2's own pattern cuts this at every diacritic (D-014); marks belong to the word.
    word = "مُحَمَّد"
    assert pretokenize(word) == [word]
    assert pretokenize(f"قال {word} ذلك") == ["قال", f" {word}", " ذلك"]


def test_numbers_and_symbols_are_separate_pieces():
    assert pretokenize("في 2024، و٣٤٥ ؟") == ["في", " 2024", "،", " و", "٣٤٥", " ؟"]


def test_whitespace_runs_keep_the_last_space_for_the_next_word():
    assert pretokenize("a  b") == ["a", " ", " b"]


def test_a_paragraph_break_before_a_word_is_two_pieces_as_in_gpt2():
    # GPT-2's `\s+(?!\S)` leaves the last whitespace character for the next piece, and only a
    # space (not a newline) can be glued to a word. Changing it would save 1.9% of wiki_ar's
    # pieces and nothing in the other sources (D-014), so the paper's behaviour is kept.
    assert pretokenize("a\n\nb") == ["a", "\n", "\n", "b"]
    assert pretokenize("a\n\n") == ["a", "\n\n"]


def test_empty():
    assert pretokenize("") == []


@pytest.mark.parametrize("seed", range(5))
def test_pieces_always_join_back_to_the_text(seed: int):
    alphabet = list("ab 1٣\n\t.,'!؟،") + ["م", "ُ", "ّ", "ـ", "é", "😀", "\u200f", "ﻻ", " "]
    rng = random.Random(seed)
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        assert "".join(pretokenize(text)) == text, repr(text)
