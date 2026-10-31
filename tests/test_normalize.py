import random

import pytest

from scratchllm.tokenizer.normalize import normalize

# ---------- presentation forms ----------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("ﻻ", "لا"),  # the lam-alef ligature is two letters
        ("ﷲ", "الله"),  # one code point for a whole word
        ("ﷺ", "صلى الله عليه وسلم"),
        ("ﺍﻟﻣﺎ", "الما"),  # isolated/initial/initial/final glyph forms
    ],
)
def test_presentation_forms_become_the_ordinary_letters(before: str, after: str):
    assert normalize(before) == after


def test_the_tatweel_inside_a_presentation_form_is_removed_too_so_nfkc_runs_first():
    # NFKC of the fatha medial form is tatweel + fatha. Removing the tatweel BEFORE NFKC would
    # leave a stray tatweel; removing it after leaves the fatha alone.
    assert normalize("ﹷ") == "َ"
    assert normalize("مﹷ") == "مَ"


def test_nfkc_is_not_applied_outside_the_presentation_blocks():
    # English text keeps its symbols; NFKC would have changed every one of these.
    text = "area 5 m² … Acme™ ½ ﬁnd Ａ"
    assert normalize(text) == text


# ---------- what is deliberately left alone (D-006) ----------


@pytest.mark.parametrize(
    "text",
    [
        "٣ و ٤٥ و ۱۲",  # Arabic-Indic and extended digits
        "مُحَمَّدٌ",  # diacritics, shadda included
        "أإآا ىي ةه",  # alef, yeh and teh-marbuta spellings
        "پ چ گ ژ",  # Persian letters
    ],
)
def test_nothing_that_changes_a_word_is_touched(text: str):
    assert normalize(text) == text


# ---------- tatweel ----------


def test_the_tatweel_is_removed():
    assert normalize("مــــرحبا") == "مرحبا"
    assert normalize("ـ") == ""


# ---------- whitespace ----------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("a  b", "a b"),
        ("a\tb", "a b"),
        ("a b", "a b"),  # no-break space
        ("a   \t b", "a b"),
        ("a\r\nb", "a\nb"),
        ("a \n b", "a\nb"),
        ("a\n\n\n\nb", "a\n\nb"),
        ("a\n\nb", "a\n\nb"),  # a paragraph break stays
        ("  a b  \n", "a b"),
        ("\n\n a", "a"),
    ],
)
def test_whitespace(before: str, after: str):
    assert normalize(before) == after


def test_newlines_survive_as_line_structure():
    assert normalize("سطر أول\nسطر ثان\n\nفقرة") == "سطر أول\nسطر ثان\n\nفقرة"


def test_empty_and_blank_text():
    assert normalize("") == ""
    assert normalize(" \n\t ") == ""


# ---------- properties ----------


def test_clean_text_is_unchanged():
    text = "ذهب الولد إلى المدرسة.\nAnd then he came back."
    assert normalize(text) == text


def test_normalize_is_idempotent_on_random_mixed_text():
    alphabet = list("ab ا ب م ـ \t\n \r") + ["ﻻ", "ﹷ", "ﷲ", "²", "٣", "َ"]
    rng = random.Random(0)
    for _ in range(2000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        once = normalize(text)
        assert normalize(once) == once, repr(text)


def test_output_has_no_tatweel_no_presentation_form_and_no_odd_whitespace():
    alphabet = list("ab ا م ـ \t\n \r") + ["ﻻ", "ﹷ", "ﷲ", "ﺍ"]
    rng = random.Random(1)
    for _ in range(2000):
        out = normalize("".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40))))
        assert "ـ" not in out and " " not in out and "\t" not in out
        assert not any(0xFB50 <= ord(c) <= 0xFDFF or 0xFE70 <= ord(c) <= 0xFEFF for c in out)
        assert "  " not in out and "\n\n\n" not in out and " \n" not in out and "\n " not in out
