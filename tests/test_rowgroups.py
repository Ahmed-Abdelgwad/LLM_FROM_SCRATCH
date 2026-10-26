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
    FooterCache,
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
    data = tmp_path / "corpus"
    data.mkdir()
    for prefix in ("a", "b", "c"):
        make_parquet(data / f"{prefix}.parquet", prefix, 100, 10)
    return data


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "footer_cache"


class SpyFS:
    """Local filesystem that records which threads opened files and can inject failures."""

    def __init__(self, fail=None):
        self.inner = LocalFileSystem()
        self.fail = fail or (lambda path, n: False)
        self.opens: Counter[str] = Counter()
        self.threads: set[int] = set()
        self._lock = threading.Lock()

    def glob(self, path: str, **kwargs: Any) -> Any:
        return self.inner.glob(path, **kwargs)

    def open(self, path: str, mode: str = "rb") -> Any:
        with self._lock:
            self.opens[path] += 1
            n = self.opens[path]
            self.threads.add(threading.get_ident())
        if self.fail(path, n):
            raise OSError("injected failure")
        return self.inner.open(path, mode)


def prepare(fs, corpus: Path, cache_dir: Path, keep: int = 32):
    footers = FooterCache(fs, cache_dir, retry_delay=0.0, keep=keep)
    return footers, list_row_groups(fs, f"{corpus}/*.parquet", footers)


def sample(fs, footers, groups, **kw):
    kw.setdefault("rows_per_group", 3)
    kw.setdefault("seed", 0)
    kw.setdefault("retry_delay", 0.0)
    stats: Counter[str] = Counter()
    order = plan_order(groups, kw["seed"])
    rows = list(iter_sampled_rows(fs, footers, order, COLUMNS, stats=stats, **kw))
    return rows, stats


# ---------- listing and planning ----------


def test_list_row_groups_counts_every_group(corpus: Path, cache_dir: Path):
    _, groups = prepare(LocalFileSystem(), corpus, cache_dir)
    assert len(groups) == 30 and sum(g.n_rows for g in groups) == 300
    assert groups == sorted(groups)
    assert {Path(g.path).name for g in groups} == {"a.parquet", "b.parquet", "c.parquet"}


def test_list_row_groups_raises_when_nothing_matches(tmp_path: Path, cache_dir: Path):
    fs = LocalFileSystem()
    with pytest.raises(FileNotFoundError):
        list_row_groups(fs, f"{tmp_path}/*.parquet", FooterCache(fs, cache_dir))


def test_plan_is_a_deterministic_permutation(corpus: Path, cache_dir: Path):
    _, groups = prepare(LocalFileSystem(), corpus, cache_dir)
    a, b = plan_order(groups, 0), plan_order(list(reversed(groups)), 0)
    assert a == b  # input order does not matter
    assert sorted(a) == sorted(groups)  # nothing lost or duplicated
    assert plan_order(groups, 1) != a  # seed changes the order


# ---------- sampling behaviour ----------


def test_takes_requested_rows_per_group_and_only_requested_columns(corpus: Path, cache_dir: Path):
    fs = SpyFS()
    footers, groups = prepare(fs, corpus, cache_dir)
    rows, stats = sample(fs, footers, groups, workers=2)
    assert len(rows) == 30 * 3 and stats["row_groups_read"] == 30
    assert all(set(r) == set(COLUMNS) for r in rows)
    per_group = Counter((r["id"][0], int(r["id"][1:]) // 10) for r in rows)
    assert set(per_group.values()) == {3} and len(per_group) == 30
    assert len({r["id"] for r in rows}) == len(rows)


def test_same_seed_same_rows_different_seed_different_rows(corpus: Path, cache_dir: Path):
    fs = LocalFileSystem()
    footers, groups = prepare(fs, corpus, cache_dir)
    first, _ = sample(fs, footers, groups, workers=4)
    again, _ = sample(fs, footers, groups, workers=4)
    other, _ = sample(fs, footers, groups, workers=4, seed=1)
    assert first == again  # parallel reads are consumed in plan order
    assert first != other


def test_a_short_prefix_of_the_plan_already_spans_several_files(corpus: Path, cache_dir: Path):
    fs = LocalFileSystem()
    footers, groups = prepare(fs, corpus, cache_dir)
    gen = iter_sampled_rows(fs, footers, plan_order(groups, 0), COLUMNS, 3, 0, Counter(), workers=2)
    rows = list(islice(gen, 12 * 3))
    assert len({r["id"][0] for r in rows}) >= 2  # a file has only 10 groups; 12 cannot fit in one


def test_thread_pool_never_exceeds_requested_workers(corpus: Path, cache_dir: Path):
    footers, groups = prepare(LocalFileSystem(), corpus, cache_dir)
    fs = SpyFS()
    sample(fs, footers, groups, workers=3)
    assert 1 <= len(fs.threads) <= 3


def test_stopping_early_is_clean(corpus: Path, cache_dir: Path):
    fs = LocalFileSystem()
    footers, groups = prepare(fs, corpus, cache_dir)
    gen = iter_sampled_rows(fs, footers, plan_order(groups, 0), COLUMNS, 3, 0, Counter())
    assert len(list(islice(gen, 5))) == 5
    gen.close()
    with pytest.raises(StopIteration):
        next(gen)


# ---------- failure handling ----------


def test_transient_failure_is_retried_and_nothing_is_lost(corpus: Path, cache_dir: Path):
    footers, groups = prepare(LocalFileSystem(), corpus, cache_dir)
    fs = SpyFS(fail=lambda path, n: path.endswith("a.parquet") and n == 1)
    rows, stats = sample(fs, footers, groups, workers=1)
    assert len(rows) == 90 and stats["row_groups_failed"] == 0


def test_a_permanently_broken_file_is_skipped_and_counted(corpus: Path, cache_dir: Path):
    footers, groups = prepare(LocalFileSystem(), corpus, cache_dir)
    fs = SpyFS(fail=lambda path, n: path.endswith("b.parquet"))
    rows, stats = sample(fs, footers, groups, workers=2, max_consecutive_failures=1000)
    assert stats["row_groups_failed"] == 10 and stats["row_groups_read"] == 20
    assert {r["id"][0] for r in rows} == {"a", "c"}


def test_gives_up_when_everything_keeps_failing(corpus: Path, cache_dir: Path):
    footers, groups = prepare(LocalFileSystem(), corpus, cache_dir)
    fs = SpyFS(fail=lambda path, n: True)
    with pytest.raises(RuntimeError, match="in a row"):
        sample(fs, footers, groups, workers=2, max_consecutive_failures=3)


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


# ---------- footer cache ----------


def test_footers_are_downloaded_once_and_reused_from_disk(corpus: Path, cache_dir: Path):
    first_fs = SpyFS()
    first, groups = prepare(first_fs, corpus, cache_dir)
    assert (first.fetched, first.hits) == (3, 0) and sum(first_fs.opens.values()) == 3

    second_fs = SpyFS()
    second, again = prepare(second_fs, corpus, cache_dir)
    assert (second.fetched, second.hits) == (0, 3)
    assert sum(second_fs.opens.values()) == 0  # no file was even opened to list the groups
    assert again == groups


def test_rows_from_cached_metadata_equal_rows_from_fresh_metadata(corpus: Path, cache_dir: Path):
    fs = LocalFileSystem()
    fresh, groups = prepare(fs, corpus, cache_dir)
    from_fresh, _ = sample(fs, fresh, groups, workers=2)
    cached, groups2 = prepare(fs, corpus, cache_dir)
    assert cached.hits == 3
    from_cache, _ = sample(fs, cached, groups2, workers=2)
    assert from_cache == from_fresh and len(from_cache) == 90


def test_memory_bound_does_not_change_results(corpus: Path, cache_dir: Path):
    fs = LocalFileSystem()
    wide, groups = prepare(fs, corpus, cache_dir, keep=32)
    narrow, groups2 = prepare(fs, corpus, cache_dir, keep=1)  # forces constant reloads from disk
    a, _ = sample(fs, wide, groups, workers=3)
    b, _ = sample(fs, narrow, groups2, workers=3)
    assert a == b and len(narrow._lru) <= 1


def test_a_damaged_cache_entry_is_replaced(corpus: Path, cache_dir: Path):
    fs = LocalFileSystem()
    prepare(fs, corpus, cache_dir)
    victim = sorted(cache_dir.glob("*.metadata"))[0]
    victim.write_bytes(b"not a parquet footer")
    footers, groups = prepare(fs, corpus, cache_dir)
    assert footers.fetched == 1 and footers.hits == 2 and len(groups) == 30
    pq.read_metadata(victim)  # rewritten with a valid footer


def test_a_changed_file_is_not_served_from_a_stale_entry(corpus: Path, cache_dir: Path):
    fs = LocalFileSystem()
    prepare(fs, corpus, cache_dir)
    make_parquet(corpus / "a.parquet", "a", 40, 10)  # same name, different size and groups
    footers, groups = prepare(fs, corpus, cache_dir)
    assert footers.fetched == 1  # only the changed file was downloaded again
    assert sum(g.n_rows for g in groups) == 40 + 100 + 100


class Recorder:
    """Wraps a file object and logs the byte ranges that are read from it."""

    def __init__(self, f: Any, log: list[tuple[int, int]]) -> None:
        self._f, self._log = f, log

    def read(self, n: int = -1) -> bytes:
        start = self._f.tell()
        data = self._f.read(n)
        self._log.append((start, start + len(data)))
        return data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._f, name)

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._f.close()


def test_reading_a_row_group_does_not_download_the_footer_again(tmp_path: Path):
    path = tmp_path / "x.parquet"
    make_parquet(path, "x", 200, 10)
    size = path.stat().st_size
    md = pq.read_metadata(path)
    footer_start = size - md.serialized_size - 8

    def footer_bytes_read(use_metadata: bool) -> int:
        log: list[tuple[int, int]] = []
        with path.open("rb") as raw:
            pf = pq.ParquetFile(Recorder(raw, log), metadata=md if use_metadata else None)
            pf.read_row_group(7, columns=COLUMNS)
        return sum(max(0, min(e, size) - max(s, footer_start)) for s, e in log)

    # control: without metadata the footer really is read, so the probe can detect it
    assert footer_bytes_read(use_metadata=False) > 0
    assert footer_bytes_read(use_metadata=True) == 0


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


def draw(corpus: Path, n_docs: int, rows_per_group: int, tmp_path: Path, fs=None):
    spec = SampleSpec(n_docs=n_docs, rows_per_group=rows_per_group, read_workers=2)
    stats: Counter[str] = Counter()
    rows = iter_source_rows(
        tiny_source(), spec, stats, fs or LocalFileSystem(), str(corpus), tmp_path / "cache"
    )
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
    assert (m.stats["footers_downloaded"], m.stats["footers_from_cache"]) == (2, 0)
    assert "short_sample" not in m.stats
    verify_manifest(manifest_path(out, "tiny"))


def test_second_draw_reuses_cached_footers_and_gives_identical_data(tmp_path: Path):
    corp = tmp_path / "corp"
    corp.mkdir()
    for prefix in ("a", "b"):
        make_parquet(corp / f"{prefix}.parquet", prefix, 200, 50)
    first, _ = draw(corp, n_docs=100, rows_per_group=250, tmp_path=tmp_path)
    second, _ = draw(corp, n_docs=100, rows_per_group=250, tmp_path=tmp_path)
    assert (second.stats["footers_downloaded"], second.stats["footers_from_cache"]) == (0, 2)
    assert second.sha256 == first.sha256  # same data, whichever way the footers arrived


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
        rows = iter_source_rows(
            src, SampleSpec(n_docs=1), Counter(), LocalFileSystem(), "/", tmp_path / "cache"
        )
        next(rows)


# ---------- timing instrumentation ----------


def slow_open(path: str, n: int) -> bool:
    time.sleep(0.05)  # every open costs 50 ms; never raises
    return False


def test_group_read_timings_are_recorded(corpus: Path, cache_dir: Path):
    footers, groups = prepare(LocalFileSystem(), corpus, cache_dir)
    _, stats = sample(SpyFS(), footers, groups, workers=2)
    assert "ms_group_reads_sum" in stats and "ms_group_read_max" in stats
    assert 0 <= stats["ms_group_read_max"] <= stats["ms_group_reads_sum"]


def test_recorded_read_time_reflects_real_slowness(corpus: Path, cache_dir: Path):
    footers, groups = prepare(LocalFileSystem(), corpus, cache_dir)
    _, stats = sample(SpyFS(fail=slow_open), footers, groups, workers=1)
    assert stats["ms_group_reads_sum"] >= 100  # 3 files opened once each at 50 ms


def test_listing_and_total_time_reach_the_manifest(tmp_path: Path):
    corp = tmp_path / "corp"
    corp.mkdir()
    for prefix in ("a", "b", "c"):
        make_parquet(corp / f"{prefix}.parquet", prefix, 100, 50)
    spec = SampleSpec(n_docs=60, rows_per_group=30, read_workers=1)
    stats: Counter[str] = Counter()
    rows = iter_source_rows(
        tiny_source(), spec, stats, SpyFS(fail=slow_open), str(corp), tmp_path / "cache"
    )
    docs = build_docs(rows, tiny_source(), spec, stats)
    m = write_sample(docs, tmp_path / "out", tiny_source(), spec, "rev0", stats)
    assert m.stats["ms_listing"] >= 150  # 3 cold footers, 1 worker, 50 ms each
    assert m.stats["ms_draw_total"] >= m.stats["ms_listing"]


class RecordingFS:
    """Local filesystem whose files log every byte range read, per path."""

    def __init__(self) -> None:
        self.inner = LocalFileSystem()
        self.logs: dict[str, list[tuple[int, int]]] = {}
        self._lock = threading.Lock()

    def glob(self, path: str, **kwargs: Any) -> Any:
        return self.inner.glob(path, **kwargs)

    def open(self, path: str, mode: str = "rb") -> Any:
        with self._lock:
            log = self.logs.setdefault(path, [])
        return Recorder(self.inner.open(path, mode), log)


def test_our_reader_never_downloads_a_footer_during_reads(corpus: Path, cache_dir: Path):
    footers, groups = prepare(LocalFileSystem(), corpus, cache_dir)  # footers fetched here
    fs = RecordingFS()
    rows, _ = sample(fs, footers, groups, workers=3)
    assert len(rows) == 90 and fs.logs  # files really were read through the recording wrapper
    for path, log in fs.logs.items():
        size = Path(path).stat().st_size
        footer_start = size - pq.read_metadata(path).serialized_size - 8
        footer_bytes = sum(max(0, min(e, size) - max(s, footer_start)) for s, e in log)
        assert footer_bytes == 0, f"{Path(path).name}: footer was downloaded again during reads"
        assert any(e > s for s, e in log)  # and data was read
