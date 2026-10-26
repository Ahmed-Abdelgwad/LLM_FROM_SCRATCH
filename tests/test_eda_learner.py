"""Tests for the Arabic-text EDA checks. Every expected number below can be counted by hand."""

from collections import Counter
from pathlib import Path

import pytest

from scratchllm.data.sampling import SampleSpec, manifest_path, text_hash, write_sample
from scratchllm.data.sources import get_source
from scratchllm.eda.checks_learner import ScriptMix, SurfaceVariation, letter_script
from scratchllm.eda.runner import analyze_source


def doc(text: str) -> dict:
    return {"id": "x", "source": "s", "url": None, "split": "train",
            "text_sha256": text_hash(text), "text": text}  # fmt: skip


def six_english_docs_manifest(tmp_path: Path) -> Path:
    src = get_source("fw2_arz")
    docs = [doc(f"english document number {i} with enough words") for i in range(6)]
    for i, d in enumerate(docs):
        d["id"] = str(i)
    out = tmp_path / "s"
    write_sample(iter(docs), out, src, SampleSpec(n_docs=len(docs)), "rev0", Counter())
    return manifest_path(out, src.name)


def run(*texts: str) -> dict:
    check = ScriptMix()
    for t in texts:
        check.update(doc(t))
    return check.result()


# ---------- what counts as a letter, and of which script ----------


@pytest.mark.parametrize(
    "ch,expected",
    [
        ("ا", "arabic"),
        ("ء", "arabic"),
        ("ﻻ", "arabic"),  # presentation-form ligature U+FEFB, from copy-pasted PDFs
        ("ﷲ", "arabic"),  # U+FDF2
        ("پ", "arabic"),  # Persian letter: Arabic *script*
        ("a", "latin"),
        ("Z", "latin"),
        ("é", "latin"),
        ("ß", "latin"),
        ("я", "other"),  # Cyrillic
        ("你", "other"),  # CJK
        ("1", None),
        ("٣", None),  # Arabic-Indic digit
        (" ", None),
        ("،", None),  # Arabic comma
        ("!", None),
        ("َ", None),  # fatha: a combining mark, not a letter
        ("ّ", None),  # shadda
        ("ـ", None),  # tatweel: isalpha() is True but we do not count it
    ],
)
def test_letter_script(ch: str, expected: str | None):
    assert letter_script(ch) == expected


# ---------- per-document shares ----------


def test_pure_arabic_pure_latin_and_half_half():
    assert run("مرحبا")["mean_arabic_share"] == 1.0
    assert run("hello")["mean_latin_share"] == 1.0
    half = run("مرحبا hello")  # 5 Arabic letters + 5 Latin letters
    assert (half["mean_arabic_share"], half["mean_latin_share"], half["mean_other_share"]) == (
        0.5,
        0.5,
        0.0,
    )


def test_other_scripts_go_to_other():
    r = run("привет 你好")  # 6 Cyrillic + 2 CJK letters
    assert r["mean_other_share"] == 1.0 and r["letters_total"] == {
        "arabic": 0,
        "latin": 0,
        "other": 8,
    }


def test_diacritics_and_tatweel_do_not_change_the_letter_count():
    plain = run("مرحبا")
    vowelled = run("مَرْحَبًا")  # same 5 letters plus fatha, sukun, fatha, tanween
    stretched = run("مــرحبا")  # same 5 letters plus two tatweels
    assert plain["letters_total"] == vowelled["letters_total"] == stretched["letters_total"]
    assert plain["letters_total"]["arabic"] == 5


def test_digits_punctuation_and_spaces_are_ignored():
    r = run("abc 123 ١٢٣ !!! ، ؟ ... ab")
    assert r["letters_total"] == {"arabic": 0, "latin": 5, "other": 0}


# ---------- documents without letters ----------


def test_documents_without_letters_are_counted_and_left_out_of_the_means():
    r = run("ااااا", "123 456", "", "   ")
    assert (r["docs"], r["docs_without_letters"], r["docs_with_letters"]) == (4, 3, 1)
    assert r["mean_arabic_share"] == 1.0  # averaged over the one document that has letters


def test_no_documents_at_all():
    r = ScriptMix().result()
    assert r["docs"] == 0 and r["mean_arabic_share"] == 0.0 and r["docs_mixed"] == 0


# ---------- thresholds are inclusive and exact ----------


def test_arabic_share_threshold_0_8():
    assert run("ا" * 4 + "a")["docs_arabic_share_ge_0_8"] == 1  # exactly 0.80
    assert run("ا" * 79 + "a" * 21)["docs_arabic_share_ge_0_8"] == 0  # 0.79
    assert run("ا" * 81 + "a" * 19)["docs_arabic_share_ge_0_8"] == 1  # 0.81


def test_latin_share_threshold_0_9():
    assert run("a" * 9 + "ا")["docs_latin_share_ge_0_9"] == 1  # exactly 0.90
    assert run("a" * 89 + "ا" * 11)["docs_latin_share_ge_0_9"] == 0  # 0.89


def test_mixed_needs_at_least_a_fifth_of_each():
    assert run("ا" + "aaaa")["docs_mixed"] == 1  # 0.2 / 0.8
    assert run("ااااا" + "a")["docs_mixed"] == 0  # 0.833 / 0.167
    assert run("ا" + "aaaaa")["docs_mixed"] == 0  # 0.167 / 0.833
    assert run("مرحبا hello")["docs_mixed"] == 1


def test_a_borderline_share_that_floats_would_misjudge():
    # 0.8 as a float is slightly above 4/5 in some computations; integers have no such edge
    for n in range(1, 40):
        assert run("ا" * (4 * n) + "a" * n)["docs_arabic_share_ge_0_8"] == 1


# ---------- aggregation ----------


def test_means_are_per_document_not_pooled():
    r = run("ا" * 100, "a")  # a long Arabic document and a one-letter Latin one
    assert r["mean_arabic_share"] == 0.5 and r["mean_latin_share"] == 0.5
    assert r["letters_total"] == {"arabic": 100, "latin": 1, "other": 0}  # pooled view differs


def test_result_has_exactly_the_documented_keys():
    assert set(run("مرحبا")) == {
        "docs", "docs_without_letters", "docs_with_letters",
        "mean_arabic_share", "mean_latin_share", "mean_other_share", "letters_total",
        "docs_arabic_share_ge_0_8", "docs_latin_share_ge_0_9", "docs_mixed",
    }  # fmt: skip


def test_runs_inside_the_eda_runner(tmp_path: Path):
    result = analyze_source(six_english_docs_manifest(tmp_path), [ScriptMix()])
    r = result["checks"]["script_mix"]
    assert r["status"] == "ok" and r["docs"] == 6 and r["docs_latin_share_ge_0_9"] == 6


# =====================================================================================
# SurfaceVariation. Marks are written as escapes so no invisible character hides in a literal.
# =====================================================================================

FATHA, SUKUN, FATHATAN, DAMMATAN = "َ", "ْ", "ً", "ٌ"
SHADDA, TATWEEL = "ّ", "ـ"


def sv(*texts: str) -> dict:
    check = SurfaceVariation()
    for t in texts:
        check.update(doc(t))
    return check.result()


def item(result: dict, name: str) -> dict:
    return result["items"][name]


def test_vowel_marks_are_counted_without_the_shadda():
    # م fatha ر sukun ح fatha ب fathatan ا  -> 5 letters, 4 marks
    r = sv("م" + FATHA + "ر" + SUKUN + "ح" + FATHA + "ب" + FATHATAN + "ا")
    assert r["arabic_letters"] == 5
    assert item(r, "diacritics_without_shadda") == {"docs_with": 1, "per_1000_letters": 800.0}
    assert item(r, "shadda")["docs_with"] == 0


def test_shadda_is_reported_on_its_own():
    r = sv("شد" + SHADDA + "ة")  # 3 letters, one shadda, no other mark
    assert item(r, "shadda") == {"docs_with": 1, "per_1000_letters": 333.333}
    assert item(r, "diacritics_without_shadda")["docs_with"] == 0
    assert r["pair_shares"]["shadda_among_all_diacritics"] == 1.0


def test_tatweel_is_counted_and_is_not_a_letter():
    r = sv("م" + TATWEEL * 3 + "رحبا")  # 5 letters and 3 tatweels
    assert r["arabic_letters"] == 5
    assert item(r, "tatweel")["per_1000_letters"] == 600.0


def test_alef_forms():
    r = sv("أإآا")  # hamza above, hamza below, madda, plain: one each of 4 letters
    for name in ("alef_plain", "alef_hamza_above", "alef_hamza_below", "alef_madda"):
        assert item(r, name) == {"docs_with": 1, "per_1000_letters": 250.0}
    assert r["pair_shares"]["alef_plain_among_alef_forms"] == 0.25


def test_yeh_versus_alef_maqsura():
    r = sv("على في")  # ع ل ى ف ي: one maqsura, one yeh, 5 letters
    assert item(r, "yeh")["per_1000_letters"] == 200.0
    assert item(r, "alef_maqsura")["per_1000_letters"] == 200.0
    assert r["pair_shares"]["alef_maqsura_among_yeh_like"] == 0.5


@pytest.mark.parametrize(
    "text,teh_final,heh_final",
    [
        ("مدرسة", 1, 0),
        ("مدرسه", 0, 1),
        ("مدرسة" + DAMMATAN, 1, 0),  # a mark after the letter does not hide that it is final
        ("مدرسة" + TATWEEL, 1, 0),  # neither does a tatweel
        ("مدرسة، ثم", 1, 0),  # punctuation ends the word
        ("الله", 0, 1),  # a legitimate word-final heh
        ("ه", 0, 1),  # one-letter word
        ("هل مهندس", 0, 0),  # heh inside words
        ("ةب", 0, 0),  # teh marbuta followed by a letter is inside the word
        ("مدرسة2", 0, 0),  # a following digit keeps it inside the token
        ("مدرسة ومدرسه", 1, 1),
        # a vowel mark between the heh and the next letter must not make the heh look final:
        # هَذَا = heh, fatha, thal, fatha, alef  (the mark is not \w, so only stripping it is safe)
        ("\u0647\u064e\u0630\u064e\u0627", 0, 0),
        ("\u062a\u0627\u0647\u064e\u0644", 0, 0),
    ],
)
def test_word_final_teh_marbuta_and_heh(text: str, teh_final: int, heh_final: int):
    r = sv(text)
    occurrences = lambda name: round(item(r, name)["per_1000_letters"] * r["arabic_letters"] / 1000)  # noqa: E731
    assert (occurrences("teh_marbuta_final"), occurrences("heh_final")) == (teh_final, heh_final)


def test_digit_systems_are_counted_separately():
    r = sv("123 ١٢٣ ۱۲۳")  # three ASCII, three Arabic-Indic, three Extended Arabic-Indic digits
    assert r["arabic_letters"] == 0
    for name in ("digits_ascii", "digits_arabic_indic", "digits_extended_arabic_indic"):
        assert item(r, name) == {"docs_with": 1, "per_1000_letters": 0.0}  # no letters, rate is 0.0


def test_docs_with_counts_documents_while_the_rate_counts_occurrences():
    r = sv("م" + TATWEEL * 4 + "ر", "مر", "مر")  # one document has four tatweels, 6 letters in all
    assert item(r, "tatweel") == {"docs_with": 1, "per_1000_letters": round(1000 * 4 / 6, 3)}


def test_rates_use_arabic_letters_only():
    r = sv("abc defgh مرحبا" + FATHA)  # 8 Latin letters are not in the denominator
    assert r["arabic_letters"] == 5
    assert item(r, "diacritics_without_shadda")["per_1000_letters"] == 200.0


def test_empty_result_and_exact_keys():
    r = SurfaceVariation().result()
    assert set(r) == {"docs", "arabic_letters", "items", "pair_shares"}
    assert list(r["items"]) == [
        "diacritics_without_shadda", "shadda", "tatweel",
        "alef_plain", "alef_hamza_above", "alef_hamza_below", "alef_madda",
        "yeh", "alef_maqsura", "teh_marbuta_final", "heh_final",
        "digits_ascii", "digits_arabic_indic", "digits_extended_arabic_indic",
    ]  # fmt: skip
    assert all(v == {"docs_with": 0, "per_1000_letters": 0.0} for v in r["items"].values())
    assert all(v is None for v in r["pair_shares"].values())  # undefined pairs are None, not 0


def test_surface_variation_runs_inside_the_eda_runner(tmp_path: Path):
    r = analyze_source(six_english_docs_manifest(tmp_path), [SurfaceVariation()])["checks"]
    assert r["surface_variation"]["status"] == "ok" and r["surface_variation"]["docs"] == 6
