"""Streaming, seeded document sampling with split assignment and a verifiable manifest.

Selection method (recorded in every manifest, because it is not a uniform sample):
Hugging Face streaming `shuffle(seed, buffer_size)` shuffles the order of source shards and
then shuffles within a rolling buffer. It spreads the draw across shards, but documents that
sit close together inside a shard remain correlated. Treat the sample as "diverse", not "iid".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

import scratchllm
from scratchllm.data.sources import Source
from scratchllm.data.split import assign_split, split_key

SELECTION_METHOD = "hf-streaming shuffle(seed, buffer_size): shuffled shard order + rolling buffer"
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
    shuffle_buffer: int = 10_000


def text_hash(text: str) -> str:
    """Exact-duplicate fingerprint. Deliberately light: near-duplicates are an EDA question."""
    return hashlib.sha256(_WS.sub(" ", text).strip().encode("utf-8")).hexdigest()


def latest_revision(repo: str) -> str:
    """Resolve the current commit hash of a dataset repo so the sample can be re-fetched."""
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo).sha
    if sha is None:
        raise RuntimeError(f"could not resolve a revision for {repo}")
    return sha


def stream_raw(source: Source, spec: SampleSpec, revision: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        source.repo, source.config, split=source.split, streaming=True, revision=revision
    )
    yield from ds.shuffle(seed=spec.seed, buffer_size=spec.shuffle_buffer)


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
