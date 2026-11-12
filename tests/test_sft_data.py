"""The fine-tuning data rules (finetune/data.py, D-038)."""

import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scratchllm.finetune.data import (
    RAW_FILES,
    Pair,
    PairSampler,
    bucket,
    build,
    hash_unit,
    local_paths,
    prepare,
    prompt_key,
    read_split,
    script_share,
)


def pair(i, prompt, response, variety="msa", source="aya"):
    return Pair(f"{source}-{i}", source, variety, prompt, response)


# ---------- the prompt key ----------


def test_the_key_ignores_what_a_reader_ignores_and_nothing_else():
    assert prompt_key("من هو اول شهيد في غزوه احد؟") == prompt_key("من هو أول شهيد في غزوة أحد ؟")
    assert prompt_key('أجب: "كم عدد أعين الذبابة ؟".') == prompt_key("أجب كم عدد أعين الذبابة")
    assert prompt_key("مَا هُوَ") == prompt_key("ما هو")
    assert prompt_key("What  IS this?") == prompt_key("what is this")
    assert prompt_key("ما عاصمة مصر") != prompt_key("ما عاصمة تونس")


def test_script_share_counts_letters_only():
    assert script_share("القاهرة 123 !!", "ar") == 1.0
    assert script_share("SELECT name FROM t", "ar") == 0.0
    assert script_share("Paris est", "en") == 1.0
    assert script_share("12345", "en") == 0.0
    assert script_share("ab جد", "ar") == 0.5


def test_buckets_are_stable_and_have_the_intended_sizes():
    assert hash_unit("x") == hash_unit("x") and 0 <= hash_unit("x") < 1
    counts = Counter(bucket(f"prompt {i}") for i in range(20_000))
    assert abs(counts["test"] / 20_000 - 0.05) < 0.01
    assert abs(counts["val"] / 20_000 - 0.05) < 0.01


# ---------- the rules ----------


def test_empty_and_wrong_script_pairs_are_dropped():
    pool = [
        pair(1, "سؤال", "جواب عربي"),
        pair(2, "سؤال آخر", "   "),
        pair(3, "اكتب استعلام", "SELECT * FROM movies"),
        pair(4, "question", "an answer", "english"),
        pair(5, "question two", "جواب", "english"),
    ]
    out, stages = build([], pool)
    assert {p.id for p in out} == {"aya-1", "aya-4"}
    assert stages["1_nonempty"]["pool"] == {"aya/english": 2, "aya/msa": 2}
    assert stages["2_script"]["pool"] == {"aya/english": 1, "aya/msa": 1}


def test_a_pool_pair_asking_a_test_question_is_removed_even_if_spelled_differently():
    official = [pair("t1", "من هو أول شهيد في غزوة أحد؟", "حمزة")]
    pool = [
        pair(1, "من هو اول شهيد في غزوه احد", "سيد الشهداء حمزة"),  # same question
        pair(2, "من هو أول شهيد في الإسلام؟", "سمية"),  # a different one
    ]
    out, stages = build(official, pool)
    assert {p.id for p in out} == {"aya-t1", "aya-2"}
    assert stages["4_leakage_removed"] == {"aya/msa": 1}
    assert next(p for p in out if p.id == "aya-t1").split == "test"


def test_exact_duplicates_are_kept_once_but_different_answers_are_kept():
    pool = [
        pair(1, "ما عاصمة مصر؟", "القاهرة"),
        pair(2, "ما  عاصمة مصر؟ ", "القاهرة"),  # the same after normalization
        pair(3, "ما عاصمة مصر؟", "عاصمة مصر هي القاهرة"),  # a re-annotation
    ]
    out, _ = build([], pool)
    assert sorted(p.id for p in out) == ["aya-1", "aya-3"]


def test_every_pair_with_the_same_prompt_lands_in_the_same_split():
    pool = []
    for i in range(400):
        q = f"سؤال رقم {i}"
        pool += [
            pair(f"{i}a", q, "جواب أول"),
            pair(f"{i}b", q + "؟", "جواب ثان"),
            pair(f"{i}", q, "إجابة", source="cidar"),
        ]
    out, stages = build([], pool)
    splits_by_key: dict[str, set[str]] = {}
    for p in out:
        splits_by_key.setdefault(prompt_key(p.prompt), set()).add(p.split)
    assert all(len(s) == 1 for s in splits_by_key.values())
    assert {p.split for p in out} == {"train", "val", "test"}
    assert sum(stages["6_split"]["test"].values()) == sum(p.split == "test" for p in out)


# ---------- files ----------


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_prepare_reads_the_pinned_files_and_writes_splits_and_a_manifest(tmp_path):
    paths = local_paths(tmp_path / "raw")
    aya = lambda q, a, code: {"inputs": q, "targets": a, "language_code": code}  # noqa: E731
    write_parquet(
        paths["aya_train"],
        [aya(f"سؤال {i}", "جواب", "arb") for i in range(50)]
        + [aya("سؤال مغربي", "جواب", "ary"), aya("question 1", "answer", "eng")],
    )
    write_parquet(
        paths["aya_test"], [aya("سؤال 3", "جواب", "arb"), aya("question 9", "answer", "eng")]
    )
    write_parquet(paths["cidar"], [{"index": 7, "instruction": "اشرح", "output": "شرح"}])
    manifest = prepare(tmp_path / "raw", tmp_path / "out")
    everything = [p for s in ("train", "val", "test") for p in read_split(tmp_path / "out", s)]
    assert not any("مغربي" in p.prompt for p in everything)  # Moroccan is not read
    assert not any(p.prompt == "سؤال 3" and p.split != "test" for p in everything)  # leak removed
    assert {p.source for p in everything} == {"aya", "cidar"}
    assert set(manifest["inputs"]) == set(RAW_FILES) and manifest["stages"]["5_after_dedup"]
    on_disk = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["files"]["train"]["sha256"] == manifest["files"]["train"]["sha256"]
    with pytest.raises(FileNotFoundError, match="download"):
        prepare(tmp_path / "nothing", tmp_path / "out2")


# ---------- the sampler ----------


def test_the_sampler_is_a_pure_function_of_seed_step_and_micro_batch():
    pools = {"ar": list(range(100)), "en": list(range(100, 130))}
    s = PairSampler(pools, {"ar": 1.0, "en": 1.0}, 16, seed=3)
    assert s.batch(5, 1) == PairSampler(pools, {"ar": 1.0, "en": 1.0}, 16, seed=3).batch(5, 1)
    assert s.batch(5, 1) != s.batch(6, 1)
    drawn = [x for step in range(300) for x in s.batch(step, 0)]
    english = sum(x >= 100 for x in drawn) / len(drawn)
    assert abs(english - 0.5) < 0.03  # equal weights although English has 30 of 130 pairs
    with pytest.raises(ValueError, match="no training examples"):
        PairSampler({"ar": [1]}, {"ar": 1.0, "en": 1.0}, 4, 0)
