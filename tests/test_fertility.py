import json
from pathlib import Path

import pytest

from scratchllm.data.sources import get_source
from scratchllm.tokenizer.corpus import (
    EVAL_SPLITS,
    available_sources,
    prepared_texts,
    training_sources,
)
from scratchllm.tokenizer.fertility import Counts, measure
from scratchllm.tokenizer.tokenizer import Tokenizer


def test_measure_counts_what_it_says():
    counts = measure(lambda t: list(t.encode()), ["ab cd", "م"])
    assert (counts.docs, counts.chars, counts.bytes, counts.words, counts.tokens) == (2, 6, 7, 3, 7)
    assert counts.tokens_per_word == pytest.approx(7 / 3)
    assert counts.chars_per_token == pytest.approx(6 / 7)
    assert counts.bytes_per_token == pytest.approx(1.0)
    assert counts.tokens_per_1000_chars == pytest.approx(1000 * 7 / 6)


def test_measure_counts_special_tokens_apart():
    counts = measure(lambda t: [1, 2, 300, 301], ["x"], is_special=lambda i: i >= 300)
    assert counts.tokens == 4 and counts.special == 2


def test_counts_default_to_zero():
    assert Counts().tokens == 0


def test_more_merges_never_make_a_text_longer():
    texts = ["ذهب الولد إلى المدرسة ثم عاد", "the boy went to school then came back"] * 10
    full = Tokenizer.train(texts, vocab_size=256 + 60)
    previous = None
    for k in range(0, len(full.merges) + 1, 5):
        length = sum(len(Tokenizer(full.merges[:k]).encode(t)) for t in texts)
        assert previous is None or length <= previous
        previous = length


def write_docs(path: Path, docs: list[dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")


def test_prepared_texts_cleans_normalizes_and_filters_splits(tmp_path: Path):
    path = tmp_path / "fw2_arb.jsonl"
    stub = "What people are saying - Write a review Try this search over all volumes"
    write_docs(
        path,
        [
            {"split": "train", "text": "مــرحبا   بكم"},
            {"split": "val", "text": "ذهب الولد"},
            {"split": "train", "text": stub},  # dropped by the Google Books rule
            {"split": "test", "text": "ـ"},  # empty after normalization
        ],
    )
    source = get_source("fw2_arb")
    assert list(prepared_texts(path, source, splits={"train"})) == ["مرحبا بكم"]
    assert list(prepared_texts(path, source, splits=EVAL_SPLITS)) == ["ذهب الولد"]
    assert len(list(prepared_texts(path, source))) == 2


def test_prepared_texts_stops_at_the_budget(tmp_path: Path):
    path = tmp_path / "fw2_arb.jsonl"
    write_docs(path, [{"split": "train", "text": "abcdef"}] * 10)
    source = get_source("fw2_arb")
    assert len(list(prepared_texts(path, source, char_budget=13))) == 3
    assert len(list(prepared_texts(path, source, max_docs=4))) == 4


def test_training_sources_leave_out_the_evaluation_only_ones(tmp_path: Path):
    for name in ["fw2_arb", "fw2_acm", "wiki_ary", "wiki_ar", "not_a_source"]:
        (tmp_path / f"{name}.jsonl").write_text("")
    assert [s.name for s in available_sources(tmp_path)] == [
        "fw2_acm",
        "fw2_arb",
        "wiki_ar",
        "wiki_ary",
    ]
    assert [s.name for s in training_sources(tmp_path)] == ["fw2_arb", "wiki_ar"]
