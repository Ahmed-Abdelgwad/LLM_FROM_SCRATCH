import threading
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.local import LocalFileSystem

from scratchllm.data.rowgroups import (
    iter_sampled_rows,
    list_row_groups,
    plan_order,
    retry,
)
from scratchllm.data.sampling import (
    SampleSpec,
    build_docs,
    effective_rows_per_group,
    iter_source_rows,
    manifest_path,
    verify_manifest,
    write_sample,
)
from scratchllm.data.sources import Source

COLUMNS = ["id", "url", "text"]


def make_parquet(path: Path, prefix: str, n_rows: int, group_size: int) -> None:
    table = pa.table(
        {
            "id": [f"{prefix}{i}" for i in range(n_rows)],
            "url": [f"https://{prefix}.example/{i}" for i in range(n_rows)],
            "text": [f"{prefix} document number {i} " + "word " * 20 for i in range(n_rows)],
            "junk": list(range(n_rows)),  # a column we never ask for
        }
    )
    pq.write_table(table, path, row_group_size=group_size)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """3 files x 100 rows, 10 rows per row group => 30 row groups."""
    for prefix in ("a", "b", "c"):
        make_parquet(tmp_path / f"{prefix}.parquet", prefix, 100, 10)
    return tmp_path


class SpyFS:
    """Local filesystem that records which threads opened files and can inject failures."""

    def __init__(self, fail=None):
        self.inner = LocalFileSystem()
        self.fail = fail or (lambda path, n: False)
        self.opens: Counter[str] = Counter()
        self.threads: set[int] = set()
        self._lock = threading.Lock()

    def glob(self, path: str) -> list[str]:
        return self.inner.glob(path)

    def open(self, path: str, mode: str = "rb") -> Any:
        with self._lock:
            self.opens[path] += 1
            n = self.opens[path]
            self.threads.add(threading.get_ident())
        if self.fail(path, n):
            raise OSError("injected failure")
        return self.inner.open(path, mode)


def sample(fs, groups, **kw):
    kw.setdefault("rows_per_group", 3)
    kw.setdefault("seed", 0)
    kw.setdefault("retry_delay", 0.0)
    stats: Counter[str] = Counter()
    rows = list(iter_sampled_rows(fs, plan_order(groups, kw["seed"]), COLUMNS, stats=stats, **kw))
    return rows, stats


# ---------- listing and planning ----------


def test_list_row_groups_counts_every_group(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    assert len(groups) == 30 and sum(g.n_rows for g in groups) == 300
    assert groups == sorted(groups)
    assert {Path(g.path).name for g in groups} == {"a.parquet", "b.parquet", "c.parquet"}


def test_list_row_groups_raises_when_nothing_matches(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list_row_groups(LocalFileSystem(), f"{tmp_path}/*.parquet")


def test_plan_is_a_deterministic_permutation(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    a, b = plan_order(groups, 0), plan_order(list(reversed(groups)), 0)
    assert a == b  # input order does not matter
    assert sorted(a) == sorted(groups)  # nothing lost or duplicated
    assert plan_order(groups, 1) != a  # seed changes the order


# ---------- sampling behaviour ----------


def test_takes_requested_rows_per_group_and_only_requested_columns(corpus: Path):
    fs = SpyFS()
    groups = list_row_groups(fs, f"{corpus}/*.parquet")
    rows, stats = sample(fs, groups, workers=2)
    assert len(rows) == 30 * 3 and stats["row_groups_read"] == 30
    assert all(set(r) == set(COLUMNS) for r in rows)
    per_group = Counter((r["id"][0], int(r["id"][1:]) // 10) for r in rows)
    assert set(per_group.values()) == {3} and len(per_group) == 30
    assert len({r["id"] for r in rows}) == len(rows)


def test_same_seed_same_rows_different_seed_different_rows(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    first, _ = sample(LocalFileSystem(), groups, workers=4)
    again, _ = sample(LocalFileSystem(), groups, workers=4)
    other, _ = sample(LocalFileSystem(), groups, workers=4, seed=1)
    assert first == again  # parallel reads are consumed in plan order
    assert first != other


def test_a_short_prefix_of_the_plan_already_spans_several_files(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    rows = list(
        islice(
            iter_sampled_rows(
                LocalFileSystem(), plan_order(groups, 0), COLUMNS, 3, 0, Counter(), workers=2
            ),
            12 * 3,
        )
    )
    assert len({r["id"][0] for r in rows}) >= 2  # a file has only 10 groups; 12 cannot fit in one


def test_thread_pool_never_exceeds_requested_workers(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    fs = SpyFS()
    sample(fs, groups, workers=3)
    assert 1 <= len(fs.threads) <= 3


def test_stopping_early_is_clean(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    gen = iter_sampled_rows(LocalFileSystem(), plan_order(groups, 0), COLUMNS, 3, 0, Counter())
    assert len(list(islice(gen, 5))) == 5
    gen.close()
    with pytest.raises(StopIteration):
        next(gen)


# ---------- failure handling ----------


def test_transient_failure_is_retried_and_nothing_is_lost(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    fs = SpyFS(fail=lambda path, n: path.endswith("a.parquet") and n == 1)
    rows, stats = sample(fs, groups, workers=1)
    assert len(rows) == 90 and stats["row_groups_failed"] == 0


def test_a_permanently_broken_file_is_skipped_and_counted(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    fs = SpyFS(fail=lambda path, n: path.endswith("b.parquet"))
    rows, stats = sample(fs, groups, workers=2, max_consecutive_failures=1000)
    assert stats["row_groups_failed"] == 10 and stats["row_groups_read"] == 20
    assert {r["id"][0] for r in rows} == {"a", "c"}


def test_gives_up_when_everything_keeps_failing(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    fs = SpyFS(fail=lambda path, n: True)
    with pytest.raises(RuntimeError, match="in a row"):
        sample(fs, groups, workers=2, max_consecutive_failures=3)


def test_retry_helper():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("no")
        return "ok"

    assert retry(flaky, attempts=4, base_delay=0.0) == "ok" and len(calls) == 3
    calls.clear()
    with pytest.raises(OSError):
        retry(lambda: (_ for _ in ()).throw(OSError("always")), attempts=2, base_delay=0.0)


# ---------- rows-per-group sizing ----------


@pytest.mark.parametrize(
    "n_docs,n_groups,base,expected",
    [
        (50_000, 5_000, 250, 250),  # plenty of groups: unchanged
        (50_000, 100, 250, 625),  # fewer groups than needed: take more from each
        (50_000, 9, 250, 6945),  # tiny source: ask for everything (capped by the group itself)
    ],
)
def test_effective_rows_per_group(n_docs, n_groups, base, expected):
    assert effective_rows_per_group(n_docs, n_groups, base) == expected


def test_effective_rows_per_group_rejects_empty_source():
    with pytest.raises(ValueError):
        effective_rows_per_group(10, 0, 250)


# ---------- end to end on local files ----------


def tiny_source() -> Source:
    return Source("tiny", "x/y", "c", "ar", "msa", "MIT", "web", data_glob="*.parquet")


def draw(corpus: Path, n_docs: int, rows_per_group: int, tmp_path: Path):
    spec = SampleSpec(n_docs=n_docs, rows_per_group=rows_per_group, read_workers=2)
    stats: Counter[str] = Counter()
    rows = iter_source_rows(tiny_source(), spec, stats, LocalFileSystem(), str(corpus))
    docs = build_docs(rows, tiny_source(), spec, stats)
    out = tmp_path / "out"
    return write_sample(docs, out, tiny_source(), spec, "rev0", stats), out


def test_end_to_end_manifest_roundtrip(tmp_path: Path):
    corp = tmp_path / "corp"
    corp.mkdir()
    for prefix in ("a", "b"):
        make_parquet(corp / f"{prefix}.parquet", prefix, 200, 50)  # 8 groups, 400 rows
    m, out = draw(corp, n_docs=100, rows_per_group=250, tmp_path=tmp_path)
    assert m.n_docs == 100
    assert (m.stats["files_total"], m.stats["row_groups_total"], m.stats["rows_total"]) == (
        2,
        8,
        400,
    )
    assert "short_sample" not in m.stats
    verify_manifest(manifest_path(out, "tiny"))


def test_small_source_still_reaches_n_docs_by_taking_more_per_group(tmp_path: Path):
    corp = tmp_path / "corp"
    corp.mkdir()
    for prefix in ("a", "b"):
        make_parquet(corp / f"{prefix}.parquet", prefix, 200, 50)
    # base rows_per_group=10 would give at most 80 rows from 8 groups; sizing raises it
    m, _ = draw(corp, n_docs=350, rows_per_group=10, tmp_path=tmp_path)
    assert m.n_docs == 350 and m.stats["rows_per_group_effective"] >= 44


def test_source_without_data_glob_is_rejected(tmp_path: Path):
    src = Source("nog", "x/y", "c", "ar", "msa", "MIT", "web")
    with pytest.raises(ValueError, match="data_glob"):
        next(iter_source_rows(src, SampleSpec(n_docs=1), Counter(), LocalFileSystem(), "/"))


# ---------- timing instrumentation ----------


def slow_open(path: str, n: int) -> bool:
    time.sleep(0.05)  # every open costs 50 ms; never raises
    return False


def test_group_read_timings_are_recorded(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    _, stats = sample(SpyFS(), groups, workers=2)
    assert "ms_group_reads_sum" in stats and "ms_group_read_max" in stats
    assert 0 <= stats["ms_group_read_max"] <= stats["ms_group_reads_sum"]


def test_recorded_read_time_reflects_real_slowness(corpus: Path):
    groups = list_row_groups(LocalFileSystem(), f"{corpus}/*.parquet")
    _, stats = sample(SpyFS(fail=slow_open), groups, workers=1)
    assert stats["ms_group_reads_sum"] >= 100  # 3 files opened once each at 50 ms


def test_listing_and_total_time_reach_the_manifest(tmp_path: Path):
    corp = tmp_path / "corp"
    corp.mkdir()
    for prefix in ("a", "b", "c"):
        make_parquet(corp / f"{prefix}.parquet", prefix, 100, 50)
    spec = SampleSpec(n_docs=60, rows_per_group=30, read_workers=1)
    stats: Counter[str] = Counter()
    rows = iter_source_rows(tiny_source(), spec, stats, SpyFS(fail=slow_open), str(corp))
    docs = build_docs(rows, tiny_source(), spec, stats)
    m = write_sample(docs, tmp_path / "out", tiny_source(), spec, "rev0", stats)
    assert m.stats["ms_listing"] >= 150  # 3 footers, 1 worker, 50 ms each
    assert m.stats["ms_draw_total"] >= m.stats["ms_listing"]
