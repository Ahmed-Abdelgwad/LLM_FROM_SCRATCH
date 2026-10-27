from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scratchllm.cli import app
from scratchllm.data.sampling import SampleSpec, manifest_path, text_hash, write_sample
from scratchllm.data.sources import get_source
from scratchllm.eda.checks_learner import ScriptMix
from scratchllm.eda.reading import draw, is_mixed, matches, render

SRC = get_source("fw2_arz")


def doc(i: int, text: str, split: str = "train") -> dict:
    return {
        "id": str(i),
        "source": SRC.name,
        "url": None,
        "split": split,
        "text_sha256": text_hash(text),
        "text": text,
    }


MIXED = "مشاهدة وتحميل فيلم London Has Fallen مترجم"
ARABIC = "مشاهدة وتحميل فيلم الاكشن مترجم بجودة عالية"
ENGLISH = "watch and download the full movie in high quality"


@pytest.mark.parametrize("text", [MIXED, ARABIC, ENGLISH, "", "123 456", "ab" * 10 + "ب" * 4])
def test_is_mixed_agrees_with_the_script_mix_check(text: str):
    check = ScriptMix()
    check.update(doc(0, text))
    assert is_mixed(text) == bool(check.result()["docs_mixed"])


def test_kinds_select_what_they_say():
    assert matches(doc(0, MIXED), "mixed") and not matches(doc(0, ARABIC), "mixed")
    assert matches(doc(0, "قصير"), "short") and not matches(doc(0, "ك" * 300), "short")
    assert matches(doc(0, ENGLISH), "all")
    with pytest.raises(ValueError):
        matches(doc(0, ENGLISH), "nonsense")


def test_draw_is_seeded_uniform_in_size_and_reports_the_total():
    docs = [doc(i, MIXED if i % 2 else ARABIC) for i in range(100)]
    a, total = draw(iter(docs), "mixed", 10, seed=1)
    b, _ = draw(iter(docs), "mixed", 10, seed=1)
    c, _ = draw(iter(docs), "mixed", 10, seed=2)
    assert total == 50 and len(a) == 10
    assert [d["id"] for d in a] == [d["id"] for d in b]
    assert [d["id"] for d in a] != [d["id"] for d in c]
    assert all(int(d["id"]) % 2 for d in a)


def test_draw_returns_everything_when_fewer_match_than_asked():
    docs = [doc(i, MIXED) for i in range(3)]
    got, total = draw(iter(docs), "mixed", 10, seed=0)
    assert len(got) == total == 3


def test_render_shows_the_count_and_trims_text():
    out = render([doc(7, "x" * 50 + "\nsecond line")], 40, 20, "title")
    assert "1 drawn of 40" in out and "`7`" in out and "x" * 20 in out and "x" * 21 not in out


def test_cli_read_uses_the_train_split_only(tmp_path: Path):
    docs = [doc(i, MIXED + f" {i}") for i in range(5)] + [doc(99, MIXED, "val")]
    out = tmp_path / "s"
    write_sample(iter(docs), out, SRC, SampleSpec(n_docs=len(docs)), "rev0", Counter())
    result = CliRunner().invoke(
        app, ["data", "read", str(manifest_path(out, SRC.name)), "--kind", "mixed", "--n", "50"]
    )
    assert result.exit_code == 0, result.output
    assert "5 drawn of 5" in result.output and "`99`" not in result.output
