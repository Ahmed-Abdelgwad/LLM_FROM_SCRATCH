from collections import Counter
from pathlib import Path

from typer.testing import CliRunner

from scratchllm.cli import app
from scratchllm.data.cleaning import (
    BOOKS_STUB_PHRASES,
    CleaningStats,
    Rule,
    clean_docs,
    drop_books_stub,
    is_books_stub,
    rules_for,
    strip_wiki_footer,
)
from scratchllm.data.sampling import SampleSpec, manifest_path, text_hash, write_sample
from scratchllm.data.sources import get_source

HEADINGS = frozenset({"مراجع", "وصلات خارجية"})
FOOTER = "\n\nمراجع\n\nقرى ولاية سطيف\nمواليد 1951"
PROSE = "خربة بن مداني هي منطقة تابعة اقليمياً وإداريا لبلدية قجال."


def doc(i: int, text: str, source: str = "fw2_apc", split: str = "train") -> dict:
    return {
        "id": str(i),
        "source": source,
        "url": None,
        "split": split,
        "text_sha256": text_hash(text),
        "text": text,
    }


# ---------- Google Books stub ----------


def test_two_stub_sentences_make_a_stub():
    assert is_books_stub(f"{BOOKS_STUB_PHRASES[0]}\n{BOOKS_STUB_PHRASES[2]}\nا ا ا ع")


def test_one_stub_sentence_is_kept_because_the_page_may_have_a_real_description():
    # the English Google Books pages in fineweb_edu carry a blurb after the sentence
    text = f"A study of the slave trade in Virginia.\n{BOOKS_STUB_PHRASES[0]}\nUser Review"
    assert not is_books_stub(text) and drop_books_stub(text) == text


def test_ordinary_text_is_not_a_stub():
    assert not is_books_stub("انا ابلغ من العمر 18 سنة حلمت اني انجبت طفل")


def test_the_stub_rule_drops_the_document():
    assert drop_books_stub(" ".join(BOOKS_STUB_PHRASES)) is None


# ---------- Wikipedia footer ----------


def test_footer_from_the_first_heading_is_removed():
    assert strip_wiki_footer(PROSE + FOOTER, HEADINGS) == PROSE


def test_the_first_heading_is_the_cut_not_the_last():
    text = PROSE + "\n\nوصلات خارجية \n موقع\n\nمراجع\n\nتصنيف"
    assert strip_wiki_footer(text, HEADINGS) == PROSE


def test_heading_match_ignores_surrounding_spaces():
    assert strip_wiki_footer(PROSE + "\n  مراجع  \nتصنيف", HEADINGS) == PROSE


def test_a_heading_inside_a_longer_line_is_not_a_heading():
    text = PROSE + "\nانظر مراجع الكتاب في الفصل الثاني\nتصنيف"
    assert strip_wiki_footer(text, HEADINGS) == text


def test_text_without_a_footer_heading_is_unchanged():
    assert strip_wiki_footer(PROSE, HEADINGS) == PROSE


def test_a_very_long_line_after_the_heading_means_the_part_is_prose_and_is_kept():
    text = PROSE + "\n\nمراجع\n\n" + "كلمة " * 100
    assert strip_wiki_footer(text, HEADINGS) == text
    assert strip_wiki_footer(text, HEADINGS, max_line_chars=1000) == PROSE


def test_a_document_that_is_only_a_footer_is_dropped():
    assert strip_wiki_footer("\n\nمراجع\n\nتصنيف", HEADINGS) is None


def test_the_content_before_the_heading_is_untouched_including_inner_blank_lines():
    body = "الفقرة الأولى.\n\nالفقرة الثانية."
    assert strip_wiki_footer(body + FOOTER, HEADINGS) == body


# ---------- which rules apply ----------


def test_rules_follow_source_kind_and_measured_headings():
    assert [r.name for r in rules_for(get_source("fw2_apc"))] == ["books_stub"]
    assert [r.name for r in rules_for(get_source("fineweb_edu_10bt"))] == ["books_stub"]
    assert [r.name for r in rules_for(get_source("wiki_ar"))] == ["wiki_footer"]
    assert [r.name for r in rules_for(get_source("wiki_ary"))] == ["wiki_footer"]
    assert rules_for(get_source("wiki_en")) == []  # headings not measured: no guess


def test_the_wiki_rule_uses_the_headings_of_its_own_source():
    (ar,) = rules_for(get_source("wiki_ar"))
    (ary,) = rules_for(get_source("wiki_ary"))
    assert ar.run(PROSE + FOOTER) == PROSE
    assert ary.run(PROSE + FOOTER) == PROSE + FOOTER  # "مراجع" is not a wiki_ary heading
    assert ary.run(PROSE + "\n\nعيون لكلام\n\nتصنيف") == PROSE


# ---------- running the rules ----------


def test_clean_docs_counts_every_drop_and_every_shortening():
    stub = " ".join(BOOKS_STUB_PHRASES)
    rules = [
        Rule("books_stub", drop_books_stub),
        Rule("footer", lambda t: strip_wiki_footer(t, HEADINGS)),
    ]
    stats = CleaningStats()
    kept = list(
        clean_docs(
            [doc(1, stub), doc(2, PROSE + FOOTER), doc(3, PROSE)],
            rules,
            stats,
        )
    )
    assert [d["id"] for d in kept] == ["2", "3"]
    assert kept[0]["text"] == PROSE
    assert kept[0]["text_sha256"] == text_hash(PROSE + FOOTER)  # still names the raw text
    assert (stats.docs_in, stats.docs_out) == (3, 2)
    assert stats.dropped == Counter({"books_stub": 1})
    assert stats.changed == Counter({"footer": 1})
    assert stats.chars_removed["books_stub"] == len(stub)
    assert stats.chars_removed["footer"] == len(FOOTER)
    assert stats.chars_in - stats.chars_out == len(stub) + len(FOOTER)


def test_a_dropped_document_does_not_reach_later_rules():
    seen: list[str] = []
    rules = [Rule("drop", lambda t: None), Rule("spy", lambda t: seen.append(t) or t)]
    assert list(clean_docs([doc(1, "x")], rules, CleaningStats())) == []
    assert seen == []


def test_no_rules_leave_the_documents_alone():
    stats = CleaningStats()
    out = list(clean_docs([doc(1, PROSE)], [], stats))
    assert out[0]["text"] == PROSE and stats.docs_out == 1 and stats.chars_in == stats.chars_out


# ---------- the report command ----------


def test_cli_clean_report_uses_the_train_split_only(tmp_path: Path):
    src = get_source("wiki_ar")
    docs = [doc(i, PROSE + FOOTER, "wiki_ar") for i in range(3)]
    docs += [doc(99, PROSE + FOOTER, "wiki_ar", "val")]
    out = tmp_path / "s"
    write_sample(iter(docs), out, src, SampleSpec(n_docs=len(docs)), "rev0", Counter())
    result = CliRunner().invoke(app, ["data", "clean-report", str(manifest_path(out, "wiki_ar"))])
    assert result.exit_code == 0, result.output
    assert "documents 3 -> 3" in result.output
    assert "wiki_footer: dropped 0 documents, shortened 3" in result.output
