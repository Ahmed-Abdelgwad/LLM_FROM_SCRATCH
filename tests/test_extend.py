"""More training text without touching the evaluation side (data/extend.py, D-049)."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scratchllm.data.extend import build_extension, compose_tokens, evaluation_hashes
from scratchllm.data.sampling import (
    SampleSpec,
    build_docs,
    manifest_path,
    text_hash,
    verify_manifest,
    write_sample,
)
from scratchllm.data.sources import get_source
from scratchllm.data.tokens import verify_tokens

WEB = get_source("fw2_arb")
EVAL_ONLY = get_source("wiki_ary")
LONG = "نص عربي طويل بما يكفي لتجاوز الحد الأدنى للطول في هذا الاختبار. " * 2


def doc(i, split, source="fw2_arb", text=None):
    text = text or f"{LONG}{i}"
    return {"id": str(i), "source": source, "url": None, "split": split,
            "text_sha256": text_hash(text), "text": text}  # fmt: skip


def sample(folder: Path, docs, source=WEB, revision="rev0", **spec_kw) -> Path:
    spec = SampleSpec(n_docs=len(docs), **spec_kw)
    write_sample(iter(docs), folder, source, spec, revision, Counter())
    return manifest_path(folder, source.name)


# ---------- the split seed ----------


def rows(n=400):
    return [
        {"text": f"{LONG}{i}", "id": str(i), "url": f"https://site{i}.example/p"} for i in range(n)
    ]


def labels(**spec_kw):
    docs = build_docs(rows(), WEB, SampleSpec(n_docs=10_000, **spec_kw), Counter())
    return [d["split"] for d in docs]


def test_without_a_split_seed_the_split_is_the_draw_seeds_as_before():
    assert labels(seed=0) == labels(seed=0, split_seed=0)
    assert labels(seed=3) == labels(seed=3, split_seed=None)


def test_a_second_draw_with_the_first_split_seed_puts_every_site_in_the_same_split():
    assert labels(seed=1, split_seed=0) == labels(seed=0)
    assert labels(seed=1) != labels(seed=0)  # without it, sites would move between splits


# ---------- the extension ----------


@pytest.fixture
def base(tmp_path):
    docs = [doc(0, "train"), doc(1, "train"), doc(2, "val"), doc(3, "test")]
    return sample(tmp_path / "base", docs, seed=0)


def eval_set(tmp_path, base):
    other = sample(tmp_path / "base", [doc(9, "train", "wiki_ary")], source=EVAL_ONLY, seed=0)
    return evaluation_hashes([base, other])


def test_evaluation_hashes_count_every_document_of_an_evaluation_only_source(tmp_path, base):
    hashes = eval_set(tmp_path, base)
    assert hashes == {doc(2, "val")["text_sha256"], doc(3, "test")["text_sha256"],
                      doc(9, "train", "wiki_ary")["text_sha256"]}  # fmt: skip


def test_only_new_training_documents_are_kept(tmp_path, base):
    hashes = eval_set(tmp_path, base)
    draw = sample(tmp_path / "draw", [
        doc(10, "train"),                                  # new: kept
        doc(11, "train"),                                  # new: kept
        doc(1, "train"),                                   # already in the first sample
        doc(12, "val"),                                    # a validation site
        doc(13, "test"),                                   # a test site
        doc(14, "train", text=doc(9, "x")["text"]),        # duplicates an evaluation text
    ], seed=1, split_seed=0)  # fmt: skip
    receipt = build_extension(base, draw, hashes, tmp_path / "ext")
    assert receipt["kept"] == 2 and receipt["drawn"] == 6
    assert receipt["dropped_in_base_sample"] == 1 and receipt["dropped_eval_duplicate"] == 1
    assert receipt["dropped_val_site"] == 1 and receipt["dropped_test_site"] == 1
    m = verify_manifest(manifest_path(tmp_path / "ext", "fw2_arb"))
    assert m.n_docs == 2 and m.docs_per_split == {"train": 2} and "short_sample" not in m.stats
    written = [
        json.loads(x)["id"] for x in (tmp_path / "ext/fw2_arb.jsonl").read_text().splitlines()
    ]
    assert written == ["10", "11"]


@pytest.mark.parametrize(
    ("spec", "revision", "message"),
    [
        ({"seed": 1}, "rev0", "split seed"),             # split seed 1, the base's is 0
        ({"seed": 0, "split_seed": 0}, "rev0", "draw seed"),
        ({"seed": 1, "split_seed": 0}, "rev1", "revision"),
        ({"seed": 1, "split_seed": 0, "val_frac": 0.05}, "rev0", "fractions"),
    ],
)  # fmt: skip
def test_a_draw_that_could_change_the_evaluation_side_is_refused(
    tmp_path, base, spec, revision, message
):
    draw = sample(tmp_path / "draw", [doc(10, "train")], revision=revision, **spec)
    with pytest.raises(ValueError, match=message):
        build_extension(base, draw, set(), tmp_path / "ext")


def test_an_evaluation_only_source_gets_no_training_text(tmp_path):
    base = sample(tmp_path / "base", [doc(0, "train", "wiki_ary")], source=EVAL_ONLY, seed=0)
    draw = sample(tmp_path / "draw", [doc(1, "train", "wiki_ary")], source=EVAL_ONLY, seed=1,
                  split_seed=0)  # fmt: skip
    with pytest.raises(ValueError, match="evaluation-only"):
        build_extension(base, draw, set(), tmp_path / "ext")


# ---------- composing the token folder ----------


def token_dir(folder: Path, name: str, splits: dict, tokenizer="tok", rules=("pii",)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"source": name, "tokenizer_sha256": tokenizer, "vocab_size": 100, "end_of_text_id": 99,
            "dtype": "uint16", "cleaning_rules": list(rules), "input_manifest_sha256": "m",
            "splits": {}}  # fmt: skip
    for split, ids in splits.items():
        data = np.asarray(ids, dtype="<u2").tobytes()
        (folder / f"{name}.{split}.bin").write_bytes(data)
        meta["splits"][split] = {"file": f"{name}.{split}.bin", "docs": 1 if ids else 0,
                                 "chars": 3 * len(ids), "tokens": len(ids),
                                 "sha256": hashlib.sha256(data).hexdigest()}  # fmt: skip
    path = folder / f"{name}.tokens.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_composed_train_is_old_then_new_and_every_evaluation_file_is_unchanged(tmp_path):
    token_dir(tmp_path / "base", "fw2_arb", {"train": [1, 2, 99], "val": [5, 99], "test": [6, 99]})
    token_dir(tmp_path / "base", "wiki_ary", {"eval": [7, 8, 99]})
    token_dir(tmp_path / "ext", "fw2_arb", {"train": [3, 4, 99], "val": [], "test": []})
    out = tmp_path / "out"
    written = compose_tokens(tmp_path / "base", tmp_path / "ext", out)
    train = np.fromfile(out / "fw2_arb.train.bin", dtype="<u2").tolist()
    assert train == [1, 2, 99, 3, 4, 99]
    for f in ("fw2_arb.val.bin", "fw2_arb.test.bin", "wiki_ary.eval.bin"):
        assert (out / f).read_bytes() == (tmp_path / "base" / f).read_bytes()
    meta = written["fw2_arb"]
    assert meta["splits"]["train"]["tokens"] == 6 and meta["splits"]["train"]["docs"] == 2
    assert "extension" in meta and "extension" not in written["wiki_ary"]
    verify_tokens(out / "fw2_arb.tokens.json")
    verify_tokens(out / "wiki_ary.tokens.json")


def test_an_extension_made_differently_is_refused(tmp_path):
    token_dir(tmp_path / "base", "fw2_arb", {"train": [1, 99], "val": [5, 99], "test": [6, 99]})
    token_dir(
        tmp_path / "ext", "fw2_arb", {"train": [3, 99], "val": [], "test": []}, tokenizer="other"
    )
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        compose_tokens(tmp_path / "base", tmp_path / "ext", tmp_path / "out")
    token_dir(tmp_path / "ext2", "fw2_arb", {"train": [3, 99], "val": [5, 99], "test": []})
    with pytest.raises(ValueError, match="evaluation documents"):
        compose_tokens(tmp_path / "base", tmp_path / "ext2", tmp_path / "out2")
