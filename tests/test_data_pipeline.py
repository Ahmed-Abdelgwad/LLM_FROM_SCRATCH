import json
from collections import Counter
from pathlib import Path

import pytest

from scratchllm.data.leakage import collect_eval_hashes, count_train_leaks, purge_train_leaks
from scratchllm.data.sampling import (
    SampleSpec,
    build_docs,
    manifest_path,
    text_hash,
    verify_manifest,
    write_sample,
)
from scratchllm.data.sources import SOURCES, get_source
from scratchllm.data.split import assign_split, split_key

WEB = get_source("fw2_arb")
WIKI = get_source("wiki_ar")
LONG = "نص عربي طويل بما يكفي لتجاوز الحد الأدنى للطول في هذا الاختبار. " * 2


def _build(rows, source=WEB, **spec_kw):
    spec = SampleSpec(n_docs=10_000, **spec_kw)
    stats: Counter[str] = Counter()
    return list(build_docs(rows, source, spec, stats)), stats


# ---------- split ----------


def test_split_is_deterministic_and_seed_dependent():
    a = [assign_split(f"k{i}", 0, 0.1, 0.1) for i in range(500)]
    assert a == [assign_split(f"k{i}", 0, 0.1, 0.1) for i in range(500)]
    assert a != [assign_split(f"k{i}", 1, 0.1, 0.1) for i in range(500)]


def test_split_fractions_are_approximately_respected():
    c = Counter(assign_split(f"k{i}", 0, 0.1, 0.1) for i in range(20_000))
    assert 0.08 < c["val"] / 20_000 < 0.12
    assert 0.08 < c["test"] / 20_000 < 0.12


@pytest.mark.parametrize("val,test", [(-0.1, 0.1), (0.5, 0.5), (0.6, 0.6)])
def test_split_rejects_bad_fractions(val, test):
    with pytest.raises(ValueError):
        assign_split("k", 0, val, test)


def test_web_pages_of_one_domain_share_a_split():
    k1 = split_key("fw2_arb", "web", "1", "https://www.example.com/a")
    k2 = split_key("fw2_arb", "web", "2", "https://example.com/b")
    assert k1 == k2


def test_wiki_is_split_per_document_not_per_host():
    # every Wikipedia URL has the same host; domain keys would put the whole corpus in one split
    keys = {
        split_key("wiki_ar", "wiki", str(i), "https://ar.wikipedia.org/wiki/x") for i in range(50)
    }
    assert len(keys) == 50
    splits = {assign_split(k, 0, 0.2, 0.2) for k in keys}
    assert splits == {"train", "val", "test"}


def test_split_key_rejects_empty_id():
    with pytest.raises(ValueError):
        split_key("s", "wiki", "", None)


# ---------- build_docs ----------


def test_missing_ids_get_stable_distinct_content_based_ids():
    rows = [
        {"text": LONG + "A"},
        {"text": LONG + "B", "id": "  "},
        {"text": LONG + "C", "id": None},
    ]
    docs, stats = _build(rows)
    ids = [d["id"] for d in docs]
    assert len(set(ids)) == 3 and all(i.startswith("h") for i in ids)
    assert stats["synthesized_id"] == 3
    docs2, _ = _build(rows)
    assert ids == [d["id"] for d in docs2]  # stable across runs


def test_exact_duplicates_dropped_even_with_whitespace_differences():
    rows = [
        {"text": LONG, "id": "1"},
        {"text": "  " + LONG.replace(" ", "  ") + "\n", "id": "2"},
    ]
    docs, stats = _build(rows)
    assert len(docs) == 1 and stats["dropped_exact_dup"] == 1
    assert text_hash(rows[0]["text"]) == text_hash(rows[1]["text"])


def test_unusable_rows_are_counted_not_silently_lost():
    rows = [{"text": None, "id": "1"}, {"text": "قصير", "id": "2"}, {"text": LONG, "id": "3"}]
    docs, stats = _build(rows)
    assert len(docs) == 1
    assert (stats["rows_seen"], stats["dropped_no_text"], stats["dropped_short"]) == (3, 1, 1)


# ---------- manifest ----------


def _write(tmp_path, source, docs, n_docs=None):
    spec = SampleSpec(n_docs=n_docs or len(docs))
    return write_sample(iter(docs), tmp_path, source, spec, "rev0", Counter())


def _doc(i, split, text=None, source="fw2_arb"):
    text = text or f"{LONG}{i}"
    return {
        "id": str(i),
        "source": source,
        "url": None,
        "split": split,
        "text_sha256": text_hash(text),
        "text": text,
    }


def test_manifest_roundtrip_and_tamper_detection(tmp_path: Path):
    m = _write(tmp_path, WEB, [_doc(i, "train") for i in range(5)])
    assert m.n_docs == 5 and m.docs_per_split == {"train": 5}
    verify_manifest(manifest_path(tmp_path, WEB.name))

    data = tmp_path / m.file
    data.write_text(data.read_text(encoding="utf-8").replace("1", "2", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_manifest(manifest_path(tmp_path, WEB.name))


def test_short_stream_is_recorded(tmp_path: Path):
    spec = SampleSpec(n_docs=10)
    stats: Counter[str] = Counter()
    m = write_sample(iter([_doc(0, "train")]), tmp_path, WEB, spec, "rev0", stats)
    assert m.n_docs == 1 and m.stats["short_sample"] == 9


def test_manifest_json_is_valid_and_records_method(tmp_path: Path):
    _write(tmp_path, WEB, [_doc(0, "train")])
    raw = json.loads(manifest_path(tmp_path, WEB.name).read_text(encoding="utf-8"))
    assert "shuffle" in raw["selection_method"] and raw["revision"] == "rev0"


# ---------- leakage ----------


def test_cross_source_and_cross_split_leaks_are_found_and_purged(tmp_path: Path):
    shared = LONG + "shared"
    _write(tmp_path, WEB, [_doc(1, "train"), _doc(2, "val", shared)])
    _write(
        tmp_path,
        WIKI,
        [_doc(3, "train", shared, "wiki_ar"), _doc(4, "train", source="wiki_ar")],
    )
    ms = [manifest_path(tmp_path, WEB.name), manifest_path(tmp_path, WIKI.name)]

    eval_hashes = collect_eval_hashes(ms)
    assert eval_hashes == {text_hash(shared)}
    assert count_train_leaks(ms, eval_hashes) == {"wiki_ar": 1}

    out = tmp_path / "clean"
    receipts = purge_train_leaks(ms, eval_hashes, out)
    assert receipts["wiki_ar"]["removed_train_leaks"] == 1
    assert receipts["fw2_arb"]["removed_train_leaks"] == 0
    kept = [json.loads(line) for line in (out / "wiki_ar.jsonl").read_text("utf-8").splitlines()]
    assert [d["id"] for d in kept] == ["4"]
    # the eval document itself is untouched
    web_clean = [json.loads(x) for x in (out / "fw2_arb.jsonl").read_text("utf-8").splitlines()]
    assert {d["split"] for d in web_clean} == {"train", "val"}


def test_leak_analysis_refuses_files_that_drifted_from_their_manifest(tmp_path: Path):
    m = _write(tmp_path, WEB, [_doc(1, "train")])
    (tmp_path / m.file).write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        collect_eval_hashes([manifest_path(tmp_path, WEB.name)])


# ---------- registry ----------


def test_registry_entries_are_complete():
    for name, s in SOURCES.items():
        assert s.name == name and s.license and s.kind in {"web", "wiki"}
    with pytest.raises(KeyError):
        get_source("nope")
