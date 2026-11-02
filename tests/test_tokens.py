import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from scratchllm.cli import app
from scratchllm.data.sources import get_source
from scratchllm.data.tokens import DTYPE, END_OF_TEXT, tokenize_source, verify_tokens
from scratchllm.tokenizer.corpus import prepared_docs
from scratchllm.tokenizer.tokenizer import Tokenizer

ARABIC = "ذهب الولد إلى المدرسة صباحا ثم عاد إلى البيت مساء"
ENGLISH = "the boy went to school in the morning and came back"
SPECIALS = [END_OF_TEXT, "<email>", "<phone>"]


def write(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = [json.dumps({"split": s, "text": t, "url": None}, ensure_ascii=False) for s, t in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.train([ARABIC, ENGLISH] * 10, vocab_size=256 + 40 + 3, special_tokens=SPECIALS)


def read_ids(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=DTYPE)


def test_a_training_source_gets_three_files_and_documents_end_with_the_end_id(
    tok: Tokenizer, tmp_path: Path
):
    jsonl = tmp_path / "fw2_arb.jsonl"
    write(jsonl, [("train", ARABIC), ("train", ENGLISH), ("val", ARABIC), ("test", ENGLISH)])
    meta = tokenize_source(tok, "abc", get_source("fw2_arb"), jsonl, tmp_path / "out")
    assert list(meta["splits"]) == ["train", "val", "test"]
    end = tok.special_id(END_OF_TEXT)
    train = read_ids(tmp_path / "out" / "fw2_arb.train.bin")
    assert train.dtype == np.dtype("<u2")
    assert list(train).count(end) == 2 and train[-1] == end
    # cut at the end ids and decode: the documents come back exactly
    docs = np.split(train, np.flatnonzero(train == end) + 1)[:-1]
    assert [tok.decode(d[:-1].tolist()) for d in docs] == [ARABIC, ENGLISH]


def test_the_description_counts_what_the_files_hold(tok: Tokenizer, tmp_path: Path):
    jsonl = tmp_path / "fw2_arb.jsonl"
    write(jsonl, [("train", ARABIC)] * 3 + [("val", ENGLISH)])
    meta = tokenize_source(tok, "abc", get_source("fw2_arb"), jsonl, tmp_path / "out")
    train, val, test = (meta["splits"][s] for s in ("train", "val", "test"))
    per_doc = len(tok.encode(ARABIC)) + 1
    assert (train["docs"], train["tokens"], train["chars"]) == (3, 3 * per_doc, 3 * len(ARABIC))
    assert val["docs"] == 1 and test["docs"] == 0 and test["tokens"] == 0
    assert meta["tokenizer_sha256"] == "abc" and meta["vocab_size"] == tok.vocab_size
    assert meta["cleaning_rules"] == ["wikipedia_page", "books_stub", "pii"]
    on_disk = json.loads((tmp_path / "out" / "fw2_arb.tokens.json").read_text())
    assert on_disk == meta


def test_an_evaluation_only_source_gets_one_file_with_every_document(
    tok: Tokenizer, tmp_path: Path
):
    jsonl = tmp_path / "fw2_acm.jsonl"
    write(jsonl, [("train", ARABIC), ("val", ARABIC), ("test", ARABIC)])
    meta = tokenize_source(tok, "abc", get_source("fw2_acm"), jsonl, tmp_path / "out")
    assert list(meta["splits"]) == ["eval"] and meta["role"] == "evaluation-only"
    assert meta["splits"]["eval"]["docs"] == 3
    assert not list((tmp_path / "out").glob("*.train.bin"))


def test_cleaning_and_normalization_are_applied_before_tokenizing(tok: Tokenizer, tmp_path: Path):
    jsonl = tmp_path / "fw2_arb.jsonl"
    rows = [
        {"split": "train", "text": "مــرحبا   بكم", "url": None},
        {"split": "train", "text": ENGLISH, "url": "https://ar.wikipedia.org/wiki/x"},  # dropped
    ]
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    tokenize_source(tok, "abc", get_source("fw2_arb"), jsonl, tmp_path / "out")
    train = read_ids(tmp_path / "out" / "fw2_arb.train.bin")
    assert tok.decode(train[:-1].tolist()) == "مرحبا بكم"


def test_a_label_that_is_not_a_split_is_refused_and_leaves_no_files(tok: Tokenizer, tmp_path: Path):
    jsonl = tmp_path / "fw2_arb.jsonl"
    write(jsonl, [("dev", ARABIC)])
    with pytest.raises(ValueError, match="split label"):
        tokenize_source(tok, "abc", get_source("fw2_arb"), jsonl, tmp_path / "out")
    assert not [p for p in (tmp_path / "out").iterdir()]


def test_verify_detects_a_changed_file(tok: Tokenizer, tmp_path: Path):
    jsonl = tmp_path / "fw2_arb.jsonl"
    write(jsonl, [("train", ARABIC), ("val", ENGLISH)])
    tokenize_source(tok, "abc", get_source("fw2_arb"), jsonl, tmp_path / "out")
    desc = tmp_path / "out" / "fw2_arb.tokens.json"
    assert verify_tokens(desc)["source"] == "fw2_arb"
    train = tmp_path / "out" / "fw2_arb.train.bin"
    data = bytearray(train.read_bytes())
    data[0] ^= 1
    train.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        verify_tokens(desc)
    train.write_bytes(bytes(data[:-2]))
    with pytest.raises(ValueError, match="bytes"):
        verify_tokens(desc)


def test_a_vocabulary_that_does_not_fit_in_two_bytes_is_refused(tmp_path: Path):
    big = Tokenizer([], special_tokens=[f"<s{i}>" for i in range(1 << 16)])
    with pytest.raises(ValueError, match="uint16"):
        tokenize_source(big, "abc", get_source("fw2_arb"), tmp_path / "x.jsonl", tmp_path / "out")


def test_prepared_docs_gives_the_label_of_each_kept_document(tmp_path: Path):
    jsonl = tmp_path / "fw2_arb.jsonl"
    write(jsonl, [("val", ARABIC), ("train", "ـ"), ("test", ENGLISH)])
    assert list(prepared_docs(jsonl, get_source("fw2_arb"))) == [("val", ARABIC), ("test", ENGLISH)]
    assert list(prepared_docs(jsonl, get_source("fw2_arb"), splits={"test"})) == [("test", ENGLISH)]


def test_cli_tokenize_then_verify(tok: Tokenizer, tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    write(data / "fw2_arb.jsonl", [("train", ARABIC), ("val", ARABIC), ("test", ARABIC)])
    write(data / "fw2_acm.jsonl", [("train", ENGLISH)])
    tok.save(tmp_path / "tok.json")
    runner = CliRunner()
    out = tmp_path / "tokens"
    result = runner.invoke(
        app,
        [
            "data",
            "tokenize",
            str(tmp_path / "tok.json"),
            "--out-dir",
            str(out),
            "--data-dir",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output
    descriptions = sorted(str(p) for p in out.glob("*.tokens.json"))
    assert len(descriptions) == 2
    result = runner.invoke(app, ["data", "verify-tokens", *descriptions])
    assert result.exit_code == 0 and result.output.count("OK ") == 2, result.output
