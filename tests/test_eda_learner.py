"""Tests for the Arabic-text EDA checks. Every expected number below can be counted by hand."""

from collections import Counter
from pathlib import Path

import pytest

from scratchllm.data.sampling import SampleSpec, manifest_path, text_hash, write_sample
from scratchllm.data.sources import get_source
from scratchllm.eda.checks_learner import ScriptMix, letter_script
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
