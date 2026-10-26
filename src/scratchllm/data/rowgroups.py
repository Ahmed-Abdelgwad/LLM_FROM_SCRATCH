"""Row-group-level sampling of (remote) parquet corpora.

Why not HF streaming `shuffle`? It reads the source sequentially and shuffles inside a rolling
buffer, so a sample is the *head* of the first shard(s), and large sources (FineWeb-2 `arb_Arab`
has 25 shards of ~4.8 GB) open many big files at once over a slow link. Here we read the parquet
footers, pick row groups uniformly at random across *all* shards, and fetch only those, one
small range request each (row groups in our sources hold ~1000 rows, a few MB).

Everything that touches the network goes through a minimal `FileSystem` (fsspec-style), so the
logic is tested on local parquet files.
"""

from __future__ import annotations

import random
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import pyarrow.parquet as pq

T = TypeVar("T")


class FileSystem(Protocol):
    def glob(self, path: str) -> list[str]: ...
    def open(self, path: str, mode: str = "rb") -> Any: ...


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


def list_row_groups(
    fs: FileSystem, pattern: str, workers: int = 8, retry_delay: float = 1.0
) -> list[RowGroupRef]:
    """Read every matching file's footer (small) and return all its row groups, sorted."""
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet files match {pattern!r}")

    def footer(path: str) -> list[RowGroupRef]:
        def read() -> list[RowGroupRef]:
            with fs.open(path, "rb") as f:
                md = pq.ParquetFile(f).metadata
            return [
                RowGroupRef(path, i, md.row_group(i).num_rows) for i in range(md.num_row_groups)
            ]

        return retry(read, base_delay=retry_delay)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sorted(g for part in pool.map(footer, paths) for g in part)


def plan_order(groups: list[RowGroupRef], seed: int) -> list[RowGroupRef]:
    """Deterministic uniform random order over all row groups of the source."""
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    return order


def iter_sampled_rows(
    fs: FileSystem,
    order: list[RowGroupRef],
    columns: list[str],
    rows_per_group: int,
    seed: int,
    stats: Counter[str],
    workers: int = 8,
    retry_delay: float = 1.0,
    max_consecutive_failures: int = 20,
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
            cache[path] = pq.ParquetFile(fs.open(path, "rb"))
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
