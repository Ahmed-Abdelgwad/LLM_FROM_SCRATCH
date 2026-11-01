import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scratchllm.cli import app
from scratchllm.tokenizer.report import fertility_report, render_markdown, train_tokenizer
from scratchllm.tokenizer.tokenizer import Tokenizer

ARABIC = "ذهب الولد إلى المدرسة صباحا ثم عاد إلى البيت مساء والبنت ذهبت إلى السوق"
ENGLISH = "the boy went to school in the morning and came back home in the evening"


def write(path: Path, docs: list[tuple[str, str]]) -> None:
    rows = [{"split": split, "text": text} for split, text in docs]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    write(d / "fw2_arb.jsonl", [("train", ARABIC)] * 30 + [("val", ARABIC), ("test", ARABIC)])
    write(d / "fineweb_edu_10bt.jsonl", [("train", ENGLISH)] * 30 + [("val", ENGLISH)])
    write(d / "fw2_acm.jsonl", [("train", ARABIC)] * 3)  # evaluation-only: never trains
    return d


def test_training_uses_only_train_text_of_training_sources(data_dir: Path):
    tok, info = train_tokenizer(data_dir, vocab_size=256 + 40 + 3, chars_per_source=10_000)
    assert sorted(info["sources"]) == ["fineweb_edu_10bt", "fw2_arb"]
    assert info["sources"]["fw2_arb"]["docs"] == 30  # not the val/test documents
    assert tok.vocab_size == 256 + 40 + 3 and info["merges"] == 40


def test_the_character_budget_limits_each_source(data_dir: Path):
    _, info = train_tokenizer(data_dir, vocab_size=300, chars_per_source=len(ARABIC) * 4)
    assert info["sources"]["fw2_arb"]["docs"] == 4


def test_a_too_small_vocabulary_is_refused(data_dir: Path):
    with pytest.raises(ValueError):
        train_tokenizer(data_dir, vocab_size=100, chars_per_source=1000)


def test_the_fertility_report_measures_held_out_text_and_marks_unseen_sources(
    data_dir: Path, tmp_path: Path
):
    tok, _ = train_tokenizer(data_dir, vocab_size=256 + 60 + 3, chars_per_source=10_000)
    path = tmp_path / "tok.json"
    tok.save(path)
    report = fertility_report(
        path, data_dir, sizes=[256 + 20 + 3, 256 + 60 + 3], compare=[], max_docs=10
    )
    assert report["eval_sources"]["fw2_arb"] == {"role": "val+test", "language": "ar", "docs": 2}
    assert report["eval_sources"]["fw2_acm"]["role"].startswith("evaluation-only")
    assert report["eval_sources"]["fw2_acm"]["docs"] == 3
    small, large = report["results"]["ours-279"], report["results"]["ours-319"]
    for name in report["eval_sources"]:
        assert large["sources"][name]["tokens"] <= small["sources"][name]["tokens"]
    arb = large["sources"]["fw2_arb"]
    assert arb["words"] == 2 * len(ARABIC.split())
    assert arb["tokens_per_word"] == arb["tokens"] / arb["words"]
    assert large["summary"]["ar_tokens_per_word"] > 0 and "en_chars_per_token" in large["summary"]


def test_the_markdown_has_one_column_per_tokenizer_and_a_row_per_source(
    data_dir: Path, tmp_path: Path
):
    tok, _ = train_tokenizer(data_dir, vocab_size=300, chars_per_source=10_000)
    tok.save(tmp_path / "t.json")
    report = fertility_report(tmp_path / "t.json", data_dir, [300], [], max_docs=5)
    md = render_markdown(report)
    assert "| fw2_acm | evaluation-only (never seen) |" in md
    assert "| source | role | ours-300 |" in md


def test_cli_train_then_fertility(data_dir: Path, tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "art" / "tok.json"
    result = runner.invoke(
        app,
        [
            "tokenizer",
            "train",
            "--out",
            str(out),
            "--data-dir",
            str(data_dir),
            "--vocab-size",
            "300",
            "--chars-per-source",
            "5000",
        ],
    )
    assert result.exit_code == 0, result.output
    assert Tokenizer.load(out).vocab_size == 300
    assert json.loads(out.with_suffix(".provenance.json").read_text())["vocab_size"] == 300
    result = runner.invoke(
        app,
        [
            "tokenizer",
            "fertility",
            str(out),
            "--data-dir",
            str(data_dir),
            "--sizes",
            "300",
            "--out-dir",
            str(tmp_path / "rep"),
            "--run-id",
            "t",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "rep" / "t" / "report.json").exists()
    assert (tmp_path / "rep" / "t" / "report.md").exists()
