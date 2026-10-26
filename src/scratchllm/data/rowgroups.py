"""Row-group-level sampling of (remote) parquet corpora.

Why not HF streaming `shuffle`? It reads the source sequentially and shuffles inside a rolling
buffer, so a sample is the *head* of the first shard(s), and large sources (FineWeb-2 `arb_Arab`
has 25 shards of ~4.8 GB) open many big files at once over a slow link. Here we read the parquet
footers, pick row groups uniformly at random across *all* shards, and fetch only those, one
small range request each (row groups in our sources hold ~1000 rows, a few MB).

A footer is large for these files (~11 MB for an `arb_Arab` shard) and needed twice: to plan
(row counts) and to read (column offsets). `FooterCache` fetches each file's footer at most once,
keeps it on disk, and hands the same metadata to the readers, so no footer is downloaded again.

Everything that touches the network goes through a minimal `FileSystem` (fsspec-style), so the
logic is tested on local parquet files.
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import pyarrow.parquet as pq

T = TypeVar("T")


# fsspec's default read block is 5 MB with read-ahead, so each column read pulls ~5 MB. Measured
# on fineweb-2 arb_Arab (8 row groups per setting, interleaved): the default fetched 4.1x the bytes
# needed and took 11 s per group; 256 KB blocks fetched 1.2x and took 3.7 s (64 KB: 1.0x, 3.2 s).
DEFAULT_BLOCK_SIZE = 256 * 1024


class FileSystem(Protocol):
    def glob(self, path: str, **kwargs: Any) -> Any: ...
    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any: ...


@dataclass(frozen=True, order=True)
class RowGroupRef:
    # field order makes sorted() deterministic: by file, then by position in the file
    path: str
    index: int
    n_rows: int


def retry(fn: Callable[[], T], attempts: int = 4, base_delay: float = 1.0) -> T:
    """Call fn, retrying transient failures with exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")


class FooterCache:
    """Parquet file metadata, downloaded at most once per file and kept on disk.

    The cache key includes the file size, so a changed file is not served from a stale entry
    (files on the Hub are immutable per revision; the size guards local files). A damaged cache
    entry is discarded and fetched again. Writes are atomic. `keep` bounds the metadata held in
    memory; the rest is re-read from the local cache file.
    """

    def __init__(
        self, fs: FileSystem, cache_dir: Path, retry_delay: float = 1.0, keep: int = 32
    ) -> None:
        self.fs, self.cache_dir, self.retry_delay, self.keep = fs, cache_dir, retry_delay, keep
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0  # served from the on-disk cache
        self.fetched = 0  # downloaded from the filesystem
        self._sizes: dict[str, int] = {}
        self._lru: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def track(self, sizes: dict[str, int]) -> None:
        self._sizes.update(sizes)

    def _cache_file(self, path: str) -> Path:
        key = hashlib.sha256(f"{path}|{self._sizes[path]}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{key}.metadata"

    def get(self, path: str) -> Any:
        with self._lock:
            md = self._lru.get(path)
            if md is not None:
                self._lru.move_to_end(path)
                return md
        # Disk and network work happens outside the lock so footers download in parallel.
        cache_file, from_cache = self._cache_file(path), False
        md = None
        if cache_file.exists():
            try:
                md, from_cache = pq.read_metadata(cache_file), True
            except Exception:
                cache_file.unlink(missing_ok=True)  # damaged entry: fetch it again
        if md is None:
            md = retry(lambda: self._fetch(path), base_delay=self.retry_delay)
            self._store(cache_file, md)
        with self._lock:
            if from_cache:
                self.hits += 1
            else:
                self.fetched += 1
            self._lru[path] = md
            while len(self._lru) > self.keep:
                self._lru.popitem(last=False)
        return md

    def _fetch(self, path: str) -> Any:
        with self.fs.open(path, "rb") as f:
            return pq.ParquetFile(f).metadata

    @staticmethod
    def _store(cache_file: Path, md: Any) -> None:
        tmp = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            md.write_metadata_file(str(tmp))
            os.replace(tmp, cache_file)
        except OSError:
            tmp.unlink(missing_ok=True)  # a cache we cannot write only costs a later re-download


def list_row_groups(
    fs: FileSystem, pattern: str, footers: FooterCache, workers: int = 8
) -> list[RowGroupRef]:
    """Every row group of every file matching `pattern`, sorted."""
    files = fs.glob(pattern, detail=True)
    if not files:
        raise FileNotFoundError(f"no parquet files match {pattern!r}")
    footers.track({path: int(info["size"]) for path, info in files.items()})

    def groups_of(path: str) -> list[RowGroupRef]:
        md = footers.get(path)
        return [RowGroupRef(path, i, md.row_group(i).num_rows) for i in range(md.num_row_groups)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sorted(g for part in pool.map(groups_of, sorted(files)) for g in part)


def plan_order(groups: list[RowGroupRef], seed: int) -> list[RowGroupRef]:
    """Deterministic uniform random order over all row groups of the source."""
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    return order


def iter_sampled_rows(
    fs: FileSystem,
    footers: FooterCache,
    order: list[RowGroupRef],
    columns: list[str],
    rows_per_group: int,
    seed: int,
    stats: Counter[str],
    workers: int = 4,
    retry_delay: float = 1.0,
    max_consecutive_failures: int = 20,
    block_size: int | None = DEFAULT_BLOCK_SIZE,
) -> Iterator[dict[str, Any]]:
    """Yield up to `rows_per_group` random rows from each row group, in plan order.

    Reads are overlapped by a thread pool but consumed in order, with a bounded window so
    memory stays at about `2 * workers` row groups. Stop iterating at any time; the pool is
    released without waiting for in-flight reads.
    """
    local = threading.local()

    def open_file(path: str) -> pq.ParquetFile:
        cache: dict[str, pq.ParquetFile] = local.__dict__.setdefault("files", {})
        if path not in cache:
            if len(cache) >= 8:  # keep per-thread handles bounded
                cache.pop(next(iter(cache)))
            # passing the metadata means the footer is not downloaded again
            opened = fs.open(path, "rb", **({"block_size": block_size} if block_size else {}))
            cache[path] = pq.ParquetFile(opened, metadata=footers.get(path))
        return cache[path]

    def read(ref: RowGroupRef) -> tuple[list[dict[str, Any]], int]:
        """Rows of one row group and the wall time spent getting them (retries included), ms."""

        def once() -> list[dict[str, Any]]:
            try:
                table = open_file(ref.path).read_row_group(ref.index, columns=columns)
            except Exception:
                local.__dict__.setdefault("files", {}).pop(ref.path, None)  # reconnect next try
                raise
            rng = random.Random(f"{seed}:{ref.path}:{ref.index}")
            picks = sorted(rng.sample(range(table.num_rows), min(rows_per_group, table.num_rows)))
            return table.take(picks).to_pylist()

        started = time.perf_counter()
        rows = retry(once, base_delay=retry_delay)
        return rows, int((time.perf_counter() - started) * 1000)

    pool = ThreadPoolExecutor(max_workers=workers)
    window: deque[Future[tuple[list[dict[str, Any]], int]]] = deque()
    todo = iter(order)

    def fill() -> None:
        while len(window) < 2 * workers:
            ref = next(todo, None)
            if ref is None:
                return
            window.append(pool.submit(read, ref))

    consecutive_failures = 0
    try:
        fill()
        while window:
            future = window.popleft()
            try:
                rows, elapsed_ms = future.result()
            except Exception as exc:
                stats["row_groups_failed"] += 1
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        f"{consecutive_failures} row groups failed in a row; network down?"
                    ) from exc
                fill()
                continue
            consecutive_failures = 0
            stats["row_groups_read"] += 1
            # per-group wall times overlap across workers, so the sum can exceed elapsed time
            stats["ms_group_reads_sum"] += elapsed_ms
            stats["ms_group_read_max"] = max(stats["ms_group_read_max"], elapsed_ms)
            fill()
            yield from rows
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
