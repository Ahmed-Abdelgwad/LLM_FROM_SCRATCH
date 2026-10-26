"""Seeded document sampling with split assignment and a verifiable manifest.

Selection method (recorded in every manifest): row groups are drawn uniformly at random across
all parquet shards of a source (see rowgroups.py), then up to `rows_per_group` random rows are
taken from each. Rows from one row group are neighbours in the source file and can be
correlated (same crawl, same site), so the sample is spread out, not strictly iid.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

import scratchllm
from scratchllm.data.rowgroups import (
    FileSystem,
    FooterCache,
    iter_sampled_rows,
    list_row_groups,
    plan_order,
    retry,
)
from scratchllm.data.sources import Source
from scratchllm.data.split import assign_split, split_key

SELECTION_METHOD = (
    "uniform random row groups across all parquet shards (seeded), "
    "up to rows_per_group random rows from each"
)
OVERSAMPLE = 1.25  # margin for rows later dropped (short, duplicate) so n_docs is still reached
HASH_NORMALIZATION = "sha256 of utf-8 text after collapsing whitespace runs and stripping"
ID_POLICY = "missing/blank source id -> 'h' + first 16 hex of text hash; counted in stats"

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class SampleSpec:
    n_docs: int
    seed: int = 0
    val_frac: float = 0.02
    test_frac: float = 0.02
    min_chars: int = 50  # drops empty/trivial rows only; real quality filtering comes after EDA
    rows_per_group: int = 250  # rows taken from each sampled row group (raised for small sources)
    read_workers: int = 4  # best of 1/4/16 measured on fineweb-2 arb_Arab; more did not help


def text_hash(text: str) -> str:
    """Exact-duplicate fingerprint. Deliberately light: near-duplicates are an EDA question."""
    return hashlib.sha256(_WS.sub(" ", text).strip().encode("utf-8")).hexdigest()


def latest_revision(repo: str, retry_delay: float = 2.0) -> str:
    """Resolve the current commit hash of a dataset repo so the sample can be re-fetched.

    This is the first network call of every draw, so a momentary DNS or connection failure here
    used to end the whole run; it is retried like the row-group reads are.
    """
    from huggingface_hub import HfApi

    sha = retry(lambda: HfApi().dataset_info(repo).sha, base_delay=retry_delay)
    if sha is None:
        raise RuntimeError(f"could not resolve a revision for {repo}")
    return sha


def open_hf(source: Source, revision: str) -> tuple[FileSystem, str]:
    """Filesystem and root path for a pinned revision of the source's Hugging Face repo."""
    from huggingface_hub import HfFileSystem

    return HfFileSystem(), f"datasets/{source.repo}@{revision}"


def effective_rows_per_group(n_docs: int, n_groups: int, base: int) -> int:
    """Rows to take per row group so that `n_docs` can be reached from `n_groups` groups.

    With enough groups this is just `base`. A small source (few groups) would otherwise stop far
    short of `n_docs`, so we take more rows from each group, spreading evenly over all of them.
    """
    if n_groups < 1:
        raise ValueError("source has no row groups")
    groups_needed = math.ceil(OVERSAMPLE * n_docs / base)
    if n_groups >= groups_needed:
        return base
    return math.ceil(OVERSAMPLE * n_docs / n_groups)


def iter_source_rows(
    source: Source,
    spec: SampleSpec,
    stats: Counter[str],
    fs: FileSystem,
    root: str,
    cache_dir: Path,
) -> Iterator[dict[str, Any]]:
    """Lazily yield sampled raw rows of `source`; records what it found in `stats`.

    `cache_dir` holds the parquet footers (downloaded once per file, reused by later draws).
    """
    if not source.data_glob:
        raise ValueError(f"source {source.name!r} has no data_glob")
    started = time.perf_counter()
    footers = FooterCache(fs, cache_dir)
    groups = list_row_groups(fs, f"{root}/{source.data_glob}", footers, workers=spec.read_workers)
    stats["ms_listing"] = int((time.perf_counter() - started) * 1000)
    stats["footers_from_cache"] = footers.hits  # counted during listing only
    stats["footers_downloaded"] = footers.fetched
    per_group = effective_rows_per_group(spec.n_docs, len(groups), spec.rows_per_group)
    stats["files_total"] = len({g.path for g in groups})
    stats["row_groups_total"] = len(groups)
    stats["rows_total"] = sum(g.n_rows for g in groups)
    stats["rows_per_group_effective"] = per_group
    columns = [c for c in (source.text_field, source.id_field, source.url_field) if c]
    yield from iter_sampled_rows(
        fs,
        footers,
        plan_order(groups, spec.seed),
        columns,
        per_group,
        spec.seed,
        stats,
        workers=spec.read_workers,
    )


def build_docs(
    rows: Iterable[dict[str, Any]], source: Source, spec: SampleSpec, stats: Counter[str]
) -> Iterator[dict[str, Any]]:
    """Normalize raw rows, drop unusable/duplicate ones, label splits. Network-free.

    `stats` is filled in as rows are consumed, so every drop is accounted for.
    """
    seen: set[str] = set()
    for row in rows:
        stats["rows_seen"] += 1
        text = row.get(source.text_field)
        if not isinstance(text, str):
            stats["dropped_no_text"] += 1
            continue
        if len(text) < spec.min_chars:
            stats["dropped_short"] += 1
            continue
        digest = text_hash(text)
        if digest in seen:
            stats["dropped_exact_dup"] += 1
            continue
        seen.add(digest)

        raw_id = row.get(source.id_field)
        doc_id = str(raw_id).strip() if raw_id is not None else ""
        if not doc_id:
            doc_id = "h" + digest[:16]
            stats["synthesized_id"] += 1
        url = row.get(source.url_field) if source.url_field else None
        url = url if isinstance(url, str) and url else None

        key = split_key(source.name, source.kind, doc_id, url)
        yield {
            "id": doc_id,
            "source": source.name,
            "url": url,
            "split": assign_split(key, spec.seed, spec.val_frac, spec.test_frac),
            "text_sha256": digest,
            "text": text,
        }


@dataclass
class Manifest:
    source: dict[str, Any]
    revision: str
    spec: dict[str, Any]
    selection_method: str
    hash_normalization: str
    id_policy: str
    stats: dict[str, int]
    n_docs: int
    docs_per_split: dict[str, int]
    n_chars: int
    file: str
    sha256: str
    created_utc: str
    scratchllm_version: str


def manifest_path(out_dir: Path, source_name: str) -> Path:
    return out_dir / f"{source_name}.manifest.json"


def write_sample(
    docs: Iterable[dict[str, Any]],
    out_dir: Path,
    source: Source,
    spec: SampleSpec,
    revision: str,
    stats: Counter[str],
) -> Manifest:
    """Write up to spec.n_docs records as JSONL plus a manifest. Writes are atomic.

    Call `stats` only after this returns: it is filled lazily while `docs` is consumed.
    """
    draw_started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{source.name}.jsonl"
    tmp = final.with_suffix(".jsonl.tmp")
    sha, per_split, n_chars, n = hashlib.sha256(), Counter[str](), 0, 0
    with open(tmp, "wb") as f:
        for doc in islice(docs, spec.n_docs):
            line = (json.dumps(doc, ensure_ascii=False) + "\n").encode("utf-8")
            f.write(line)
            sha.update(line)
            per_split[doc["split"]] += 1
            n_chars += len(doc["text"])
            n += 1
        f.flush()
        os.fsync(f.fileno())
    if n < spec.n_docs:
        stats["short_sample"] = spec.n_docs - n  # the stream ended before reaching n_docs
    stats["ms_draw_total"] = int((time.perf_counter() - draw_started) * 1000)  # list+read+write
    os.replace(tmp, final)

    manifest = Manifest(
        source=asdict(source),
        revision=revision,
        spec=asdict(spec),
        selection_method=SELECTION_METHOD,
        hash_normalization=HASH_NORMALIZATION,
        id_policy=ID_POLICY,
        stats=dict(stats),
        n_docs=n,
        docs_per_split=dict(per_split),
        n_chars=n_chars,
        file=final.name,
        sha256=sha.hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        scratchllm_version=scratchllm.__version__,
    )
    manifest_path(out_dir, source.name).write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_manifest(path: Path) -> Manifest:
    """Recompute the data file's hash and counts; raise if it no longer matches its manifest."""
    manifest = Manifest(**json.loads(path.read_text(encoding="utf-8")))
    data = path.parent / manifest.file
    sha, per_split, n = hashlib.sha256(), Counter[str](), 0
    with open(data, "rb") as f:
        for line in f:
            sha.update(line)
            per_split[json.loads(line)["split"]] += 1
            n += 1
    if sha.hexdigest() != manifest.sha256:
        raise ValueError(f"{data}: sha256 differs from manifest (file modified or corrupted)")
    if n != manifest.n_docs or dict(per_split) != manifest.docs_per_split:
        raise ValueError(f"{data}: document counts differ from manifest")
    return manifest
