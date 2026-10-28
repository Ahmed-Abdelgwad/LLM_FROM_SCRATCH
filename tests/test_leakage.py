import random
from collections import Counter
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from scratchllm.cli import app
from scratchllm.data.sampling import SampleSpec, manifest_path, text_hash, write_sample
from scratchllm.data.sources import get_source
from scratchllm.eda.checks_learner import NearDuplicates
from scratchllm.eda.leakage import (
    EVAL,
    EVAL_ONLY,
    TRAIN,
    build_index,
    find_neighbours,
    hash_functions,
    jaccard,
    role_of,
    shingle_hashes,
    signature,
    summarize,
)


def words(rng: random.Random, n: int) -> list[str]:
    return [f"w{rng.randrange(10**6)}" for _ in range(n)]


def doc(i: int, text: str, source: str, split: str) -> dict:
    return {
        "id": str(i),
        "source": source,
        "url": f"https://example.org/{i}",
        "split": split,
        "text_sha256": text_hash(text),
        "text": text,
    }


# ---------- the building blocks ----------


def test_signatures_equal_the_ones_near_duplicates_makes():
    rng = random.Random(1)
    check = NearDuplicates()
    a, b = hash_functions()
    texts = [" ".join(words(rng, rng.randrange(3, 200))) for _ in range(30)]
    for i, t in enumerate(texts):
        check.update({"id": str(i), "text": t})
    mine = [signature(shingle_hashes(t), a, b) for t in texts]
    assert np.array_equal(np.stack(mine), np.stack(check._signatures))


def test_exact_jaccard():
    x, y = np.array([1, 2, 3, 4], dtype=np.uint32), np.array([3, 4, 5, 6], dtype=np.uint32)
    assert jaccard(x, y) == 2 / 6
    assert jaccard(x, x) == 1.0
    assert jaccard(x, np.empty(0, dtype=np.uint32)) == 0.0


def test_a_text_shorter_than_one_shingle_is_one_shingle_and_empty_text_has_none():
    assert shingle_hashes("one two").size == 1
    assert shingle_hashes("   ").size == 0


def test_roles_follow_the_source_and_the_split():
    assert role_of("fw2_arz", "train") == TRAIN
    assert role_of("fw2_arz", "val") == role_of("fw2_arz", "test") == EVAL
    assert role_of("fw2_acm", "train") == EVAL_ONLY  # an evaluation-only source is all evaluation


# ---------- detection against planted similarity ----------


def perturb(rng: random.Random, text: str, rate: float) -> str:
    return " ".join(w if rng.random() > rate else f"x{rng.randrange(10**9)}" for w in text.split())


def build_planted(tmp_path: Path, rates: list[float]):
    rng = random.Random(7)
    bases = [" ".join(words(rng, 120)) for _ in range(len(rates) * 15)]
    train = [doc(i, t, "fw2_arz", "train") for i, t in enumerate(bases)]
    variants = []
    for k, t in enumerate(bases):
        variants.append(doc(1000 + k, perturb(rng, t, rates[k % len(rates)]), "fw2_arz", "val"))
    unrelated = [doc(5000 + i, " ".join(words(rng, 120)), "fw2_arz", "test") for i in range(20)]
    out = tmp_path / "s"
    docs = train + variants + unrelated
    write_sample(
        iter(docs), out, get_source("fw2_arz"), SampleSpec(n_docs=len(docs)), "r", Counter()
    )
    return build_index([manifest_path(out, "fw2_arz")], apply_rules=False)


def test_planted_neighbours_are_found_in_proportion_to_their_true_similarity(tmp_path: Path):
    index = build_planted(tmp_path, rates=[0.01, 0.04, 0.3, 0.6])
    found = find_neighbours(index)
    high = low = 0
    n_high = n_low = 0
    for q, e in enumerate(index.entries):
        if e.role != EVAL or not e.doc_id.startswith("10"):
            continue
        best = found.get(q, [None])[0]
        base = index.entries[q]  # the variant; its base is the train doc with the same offset
        k = int(e.doc_id) - 1000
        truth = jaccard(index.hashes[q], index.hashes[k])  # train doc k is the base
        if truth >= 0.8:
            n_high += 1
            high += best is not None and best.similarity >= 0.8 and best.pool == k
        elif truth < 0.5:
            n_low += 1
            low += best is None or best.similarity < 0.8
        del base
    assert n_high >= 10 and high == n_high  # every true neighbour at >= 0.8 is found
    assert n_low >= 10 and low == n_low  # and nothing far below is reported as one


def test_unrelated_text_is_not_a_neighbour(tmp_path: Path):
    index = build_planted(tmp_path, rates=[0.01])
    found = find_neighbours(index)
    unrelated = [q for q, e in enumerate(index.entries) if e.doc_id.startswith("50")]
    assert unrelated and all(q not in found for q in unrelated)


def test_the_reported_similarity_is_the_exact_one(tmp_path: Path):
    index = build_planted(tmp_path, rates=[0.04])
    for matches in find_neighbours(index).values():
        for m in matches:
            assert m.similarity == jaccard(index.hashes[m.query], index.hashes[m.pool])


def test_summary_counts_exact_duplicates_and_splits_the_best_match_by_source(tmp_path: Path):
    rng = random.Random(3)
    text = " ".join(words(rng, 100))
    other = " ".join(words(rng, 100))
    a = [doc(1, text, "fw2_arz", "train"), doc(2, other, "fw2_arz", "val")]
    b = [doc(3, other + " tail", "fw2_arb", "train")]  # near-duplicate of the arz val document
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    write_sample(iter(a), out_a, get_source("fw2_arz"), SampleSpec(n_docs=2), "r", Counter())
    write_sample(iter(b), out_b, get_source("fw2_arb"), SampleSpec(n_docs=1), "r", Counter())
    index = build_index(
        [manifest_path(out_a, "fw2_arz"), manifest_path(out_b, "fw2_arb")], apply_rules=False
    )
    report = summarize(index, find_neighbours(index), bucket_cap=50)
    group = report["groups"]["fw2_arz (val+test)"]
    assert group["documents"] == 1 and group["near_0_8"] == 1
    assert group["best_in_other_source"] == 1 and group["best_in_same_source"] == 0
    assert group["exact_duplicates"] == 0


def test_cli_leakage_report_writes_files_and_removes_nothing(tmp_path: Path):
    rng = random.Random(5)
    t = " ".join(words(rng, 100))
    docs = [doc(1, t, "fw2_arz", "train"), doc(2, t, "fw2_arz", "val")]
    out = tmp_path / "s"
    write_sample(iter(docs), out, get_source("fw2_arz"), SampleSpec(n_docs=2), "r", Counter())
    m = manifest_path(out, "fw2_arz")
    before = (out / "fw2_arz.jsonl").read_bytes()
    result = CliRunner().invoke(
        app, ["data", "leakage-report", str(m), "--out-dir", str(tmp_path / "r"), "--run-id", "t"]
    )
    assert result.exit_code == 0, result.output
    text = (tmp_path / "r" / "t" / "leakage.md").read_text(encoding="utf-8")
    assert "fw2_arz (val+test)" in text and "| 1 | 1 | 1 |" in text  # 1 doc, 1 exact, 1 near
    assert (out / "fw2_arz.jsonl").read_bytes() == before  # data file untouched


def test_a_neighbour_between_0_6_and_0_8_is_counted_loosely_but_not_as_a_near_duplicate(
    tmp_path: Path,
):
    rng = random.Random(11)
    base = " ".join(words(rng, 120))
    variant = perturb(rng, base, 0.04)
    similarity = jaccard(shingle_hashes(base), shingle_hashes(variant))
    assert 0.6 <= similarity < 0.8, similarity  # the fixture itself, so the test cannot go stale
    docs = [doc(1, base, "fw2_arz", "train"), doc(2, variant, "fw2_arz", "val")]
    out = tmp_path / "s"
    write_sample(iter(docs), out, get_source("fw2_arz"), SampleSpec(n_docs=2), "r", Counter())
    index = build_index([manifest_path(out, "fw2_arz")], apply_rules=False)
    group = summarize(index, find_neighbours(index), bucket_cap=50)["groups"]["fw2_arz (val+test)"]
    assert group["near_0_6"] == 1 and group["near_0_8"] == 0


def test_totals_are_counted_and_reconcile_with_the_manifest(tmp_path: Path):
    index = build_planted(tmp_path, rates=[0.01, 0.3])
    report = summarize(index, find_neighbours(index), bucket_cap=50)
    manifest_counts = report["reconciliation"]["fw2_arz"]["manifest_docs_per_split"]
    assert report["totals"]["training_pool"] == manifest_counts["train"]
    assert (
        report["totals"]["evaluation_val_test"] == manifest_counts["val"] + manifest_counts["test"]
    )
    assert report["totals"]["evaluation_only_sources"] == 0
    assert report["all_counts_reconcile"] is True
    assert "fw2_arz" in report["sample_coverage"]


def test_a_count_that_disagrees_with_the_manifest_is_flagged(tmp_path: Path):
    index = build_planted(tmp_path, rates=[0.01])
    index.sources["fw2_arz"]["manifest_docs_per_split"]["train"] += 1  # as if a doc went missing
    report = summarize(index, find_neighbours(index), bucket_cap=50)
    assert report["all_counts_reconcile"] is False
    assert report["reconciliation"]["fw2_arz"]["manifest_matches_read"] is False


def test_documents_removed_by_the_rules_are_accounted_for(tmp_path: Path):
    stub = " ".join(["What people are saying - Write a review", "Try this search over all volumes"])
    docs = [doc(1, stub, "fw2_apc", "train"), doc(2, "نص عادي " * 30, "fw2_apc", "train")]
    out = tmp_path / "s"
    write_sample(iter(docs), out, get_source("fw2_apc"), SampleSpec(n_docs=2), "r", Counter())
    index = build_index([manifest_path(out, "fw2_apc")])  # the Google Books rule is on
    info = summarize(index, find_neighbours(index), 50)["reconciliation"]["fw2_apc"]
    assert info["removed_by_cleaning"] == 1 and info["in_index_per_split"] == {"train": 1}
    assert info["read_minus_removed_equals_in_index"] is True
