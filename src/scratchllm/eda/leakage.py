"""Near-duplicate leakage between evaluation text and training text, across all sources.

Measure only: nothing here deletes a document or rewrites a data file. Each evaluation document
is compared with every training document of every source, on the text as it would be trained
on (after the cleaning rules), and the result says how many evaluation documents have a
near-duplicate in the training pool (acceptance criteria A8 and A9) and which ones.

Why a second implementation next to `NearDuplicates`: that check answers "how much of one
sample is near-duplicate" (A2) and compares each member of an LSH bucket only with the bucket's
first member, a deliberate approximation that is fine for counting clusters but not for asking
"does THIS evaluation document have a neighbour in THAT pool". Here:

- the pool and the queries are separate sets, so a pair is only ever train-versus-evaluation;
- an LSH candidate is accepted only after the exact Jaccard similarity of the two shingle sets
  is computed, so the 128-number signature (standard error about 0.03) only finds candidates and
  never decides;
- the signatures use the same hash functions as `NearDuplicates` (a test checks this), so the two
  agree on what a signature is.

Known limit, stated in every report: a bucket holding many training documents (a template such
as "X is a village in Y") offers at most `bucket_cap` of them to each query. One accepted
neighbour is enough to flag a query, so this only loses information about how many neighbours it
has and which source the best one is in.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import scratchllm
from scratchllm.data.cleaning import CleaningStats, clean_docs, rules_for
from scratchllm.data.sampling import text_hash, verify_manifest
from scratchllm.data.sources import get_source
from scratchllm.eda.checks_learner import word_shingles
from scratchllm.eda.runner import TRACKED_PREFIX

PRIME = 4_294_967_311  # same as NearDuplicates: a*x + b < 2**64 for a, x < 2**32
NUM_PERM = 128
ROWS_PER_BAND = 8
SHINGLE_WORDS = 5
THRESHOLD = 0.8  # A9
LOOSE_THRESHOLD = 0.6  # reported too, to show how close the neighbours are
ESTIMATE_FLOOR = 0.5  # candidates below this signature agreement are not worth an exact check

# A document's role in the comparison.
TRAIN, EVAL, EVAL_ONLY = 0, 1, 2

# D-002: these sources are never trained on; fw2_apc is undecided and is treated the same here.
EVAL_ONLY_SOURCES = frozenset({"fw2_acm", "fw2_apc", "wiki_ary"})


def hash_functions(num_perm: int = NUM_PERM, seed: int = 0) -> tuple[Any, Any]:
    """The (a, b) pairs of NearDuplicates for the same seed, drawn in the same order."""
    rng = np.random.RandomState(seed)
    a = rng.randint(1, 2**32 - 1, size=num_perm, dtype=np.uint64)[:, None]
    b = rng.randint(0, 2**32 - 1, size=num_perm, dtype=np.uint64)[:, None]
    return a, b


def shingle_hashes(text: str, size: int = SHINGLE_WORDS) -> Any:
    """Sorted unique crc32 of the word shingles of `text`, as uint32 (empty if no words)."""
    shingles = word_shingles(text, size)
    if not shingles:
        return np.empty(0, dtype=np.uint32)
    values = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in shingles), np.uint64)
    return np.unique(values).astype(np.uint32)


def signature(hashes: Any, a: Any, b: Any) -> Any:
    return ((a * hashes.astype(np.uint64) + b) % PRIME).min(axis=1)


def jaccard(x: Any, y: Any) -> float:
    """Exact Jaccard similarity of two sorted unique arrays."""
    if x.size == 0 or y.size == 0:
        return 0.0
    inter = np.intersect1d(x, y, assume_unique=True).size
    return inter / (x.size + y.size - inter)


@dataclass
class Entry:
    source: str
    split: str
    doc_id: str
    role: int
    url: str | None
    start: str  # first characters of the cleaned text, for a person to read


@dataclass
class Index:
    entries: list[Entry] = field(default_factory=list)
    hashes: list[Any] = field(default_factory=list)
    exact_keys: list[str] = field(default_factory=list)
    signatures: Any = None  # (n, NUM_PERM) uint64, built by `sign`
    cleaned_away: dict[str, int] = field(default_factory=dict)  # documents the rules removed
    no_words: int = 0

    def sign(self, a: Any, b: Any) -> None:
        if not self.hashes:
            self.signatures = np.empty((0, NUM_PERM), dtype=np.uint64)
            return
        self.signatures = np.stack([signature(h, a, b) for h in self.hashes])


def role_of(source_name: str, split: str, eval_only: Iterable[str] = EVAL_ONLY_SOURCES) -> int:
    if source_name in set(eval_only):
        return EVAL_ONLY
    return TRAIN if split == "train" else EVAL


def iter_all_docs(manifest_path: Path) -> Iterator[dict[str, Any]]:
    manifest = verify_manifest(manifest_path)  # raises if the file no longer matches
    with open(manifest_path.parent / manifest.file, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_index(
    manifests: Sequence[Path],
    eval_only: Iterable[str] = EVAL_ONLY_SOURCES,
    progress: Callable[[str], None] | None = None,
    apply_rules: bool = True,
) -> Index:
    """Read every split of every sample, clean it as it would be trained on, and fingerprint it."""
    say = progress or (lambda _msg: None)
    index = Index()
    a, b = hash_functions()
    for path in manifests:
        manifest = verify_manifest(path)
        source = get_source(manifest.source["name"])
        rules = rules_for(source) if apply_rules else []
        stats = CleaningStats()
        say(f"{source.name}: reading and fingerprinting")
        n_in = 0
        for doc in clean_docs(iter_all_docs(path), rules, stats):
            n_in += 1
            text = doc["text"]
            hashes = shingle_hashes(text)
            if hashes.size == 0:
                index.no_words += 1
                continue
            index.entries.append(
                Entry(
                    source.name,
                    doc["split"],
                    str(doc["id"]),
                    role_of(source.name, doc["split"], eval_only),
                    doc.get("url"),
                    text[:120],
                )
            )
            index.hashes.append(hashes)
            index.exact_keys.append(text_hash(text))
            if n_in % 10_000 == 0:
                say(f"{source.name}: {n_in:,} documents")
        index.cleaned_away[source.name] = stats.docs_in - stats.docs_out
    say(f"signing {len(index.entries):,} documents")
    index.sign(a, b)
    return index


@dataclass
class Match:
    query: int
    pool: int
    similarity: float  # exact Jaccard of the shingle sets


def find_neighbours(
    index: Index,
    bucket_cap: int = 50,
    estimate_floor: float = ESTIMATE_FLOOR,
    rows_per_band: int = ROWS_PER_BAND,
    progress: Callable[[str], None] | None = None,
) -> dict[int, list[Match]]:
    """For every evaluation document, its training neighbours with exact Jaccard >= the floor.

    Candidates come from LSH banding over the signatures; a candidate is kept when its signature
    agreement reaches `estimate_floor`, and the number reported is the exact Jaccard.
    """
    say = progress or (lambda _msg: None)
    roles = np.array([e.role for e in index.entries])
    pool = np.flatnonzero(roles == TRAIN)
    queries = np.flatnonzero(roles != TRAIN)
    sigs = index.signatures
    candidates: dict[int, set[int]] = defaultdict(set)
    for band in range(NUM_PERM // rows_per_band):
        columns = sigs[:, band * rows_per_band : (band + 1) * rows_per_band]
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i in pool:
            buckets[columns[i].tobytes()].append(int(i))
        for qi in queries:
            members = buckets.get(columns[qi].tobytes())
            if members:
                candidates[int(qi)].update(members[:bucket_cap])
        bands = NUM_PERM // rows_per_band
        say(f"band {band + 1}/{bands}: {len(candidates):,} queries with candidates")
    found: dict[int, list[Match]] = {}
    for q, cands in candidates.items():
        ids = np.fromiter(cands, dtype=np.int64)
        agree = (sigs[ids] == sigs[q]).mean(axis=1)
        matches = [
            Match(q, int(t), jaccard(index.hashes[q], index.hashes[int(t)]))
            for t in ids[agree >= estimate_floor]
        ]
        matches = [m for m in matches if m.similarity >= estimate_floor]
        if matches:
            found[q] = sorted(matches, key=lambda m: -m.similarity)
    return found


def group_label(entry: Entry) -> str:
    kind = {EVAL: "val+test", EVAL_ONLY: "evaluation-only source"}[entry.role]
    return f"{entry.source} ({kind})"


def summarize(
    index: Index, neighbours: dict[int, list[Match]], bucket_cap: int, examples: int = 5
) -> dict[str, Any]:
    train_keys = {index.exact_keys[i] for i, e in enumerate(index.entries) if e.role == TRAIN}
    groups: dict[str, dict[str, Any]] = {}
    for q, entry in enumerate(index.entries):
        if entry.role == TRAIN:
            continue
        g = groups.setdefault(
            group_label(entry),
            {"documents": 0, "exact_duplicates": 0, "near_0_8": 0, "near_0_6": 0,
             "best_in_same_source": 0, "best_in_other_source": 0, "_examples": []},
        )  # fmt: skip
        g["documents"] += 1
        g["exact_duplicates"] += index.exact_keys[q] in train_keys
        best = neighbours.get(q, [None])[0]
        if best is None:
            continue
        g["near_0_6"] += best.similarity >= LOOSE_THRESHOLD
        if best.similarity >= THRESHOLD:
            g["near_0_8"] += 1
            same = index.entries[best.pool].source == entry.source
            g["best_in_same_source" if same else "best_in_other_source"] += 1
            g["_examples"].append((best.similarity, q, best.pool))
    for g in groups.values():
        n = g["documents"]
        g["share_near_0_8"] = round(g["near_0_8"] / n, 4) if n else 0.0
        g["share_near_0_6"] = round(g["near_0_6"] / n, 4) if n else 0.0
        g["share_exact"] = round(g["exact_duplicates"] / n, 4) if n else 0.0
        top = sorted(g.pop("_examples"), key=lambda x: (-x[0], x[1]))[:examples]
        g["examples"] = [_example(index, sim, q, p) for sim, q, p in top]
    pool_by_source: dict[str, int] = defaultdict(int)
    for e in index.entries:
        if e.role == TRAIN:
            pool_by_source[e.source] += 1
    return {
        "parameters": {
            "threshold": THRESHOLD,
            "loose_threshold": LOOSE_THRESHOLD,
            "num_perm": NUM_PERM,
            "rows_per_band": ROWS_PER_BAND,
            "shingle_words": SHINGLE_WORDS,
            "estimate_floor": ESTIMATE_FLOOR,
            "bucket_cap": bucket_cap,
        },
        "training_pool_documents": dict(pool_by_source),
        "removed_by_cleaning": index.cleaned_away,
        "documents_without_words": index.no_words,
        "groups": groups,
    }


def _example(index: Index, similarity: float, q: int, p: int) -> dict[str, Any]:
    eq, ep = index.entries[q], index.entries[p]
    return {
        "similarity": round(similarity, 3),
        "evaluation": {"source": eq.source, "split": eq.split, "id": eq.doc_id, "url": eq.url,
                       "start": eq.start},
        "training": {"source": ep.source, "split": ep.split, "id": ep.doc_id, "url": ep.url,
                     "start": ep.start},
    }  # fmt: skip


def render_markdown(report: dict[str, Any]) -> str:
    p = report["parameters"]
    lines = [
        f"# Leakage report {report['run_id']}",
        "",
        f"scratchllm {report['scratchllm_version']}. Measure only: no document was removed.",
        f"Jaccard on {p['shingle_words']}-word shingles; near-duplicate = exact Jaccard >= "
        f"{p['threshold']} (A9); {p['num_perm']} permutations, {p['rows_per_band']} rows per band; "
        f"at most {p['bucket_cap']} training documents offered per bucket.",
        "",
        "Training pool (train split, after the cleaning rules): "
        + ", ".join(f"{k} {v:,}" for k, v in report["training_pool_documents"].items()),
        "",
        "| evaluation group | documents | exact dup | near >= 0.8 | share | near >= 0.6 | "
        "best in same source | best in other source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, g in report["groups"].items():
        lines.append(
            f"| {name} | {g['documents']:,} | {g['exact_duplicates']} | {g['near_0_8']} | "
            f"{g['share_near_0_8']:.2%} | {g['near_0_6']} ({g['share_near_0_6']:.2%}) | "
            f"{g['best_in_same_source']} | {g['best_in_other_source']} |"
        )
    lines.append("")
    for name, g in report["groups"].items():
        if not g["examples"]:
            continue
        lines += [f"## {name}: closest pairs", ""]
        for ex in g["examples"]:
            ev, tr = ex["evaluation"], ex["training"]
            lines.append(f"- similarity {ex['similarity']}")
            lines.append(f"  - evaluation `{ev['id']}` {ev['url']}: {_one_line(ev.get('start'))}")
            lines.append(f"  - training {tr['source']} `{tr['id']}` {tr['url']}: "
                         f"{_one_line(tr.get('start'))}")  # fmt: skip
        lines.append("")
    return "\n".join(lines)


def _one_line(text: str | None) -> str:
    return "(text not kept in a tracked report)" if text is None else text.replace("\n", " ⏎ ")


def run_leakage(
    manifests: Sequence[Path],
    out_dir: Path,
    run_id: str | None = None,
    bucket_cap: int = 50,
    progress: Callable[[str], None] | None = None,
) -> Path:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    index = build_index(manifests, progress=progress)
    neighbours = find_neighbours(index, bucket_cap=bucket_cap, progress=progress)
    report = {
        "run_id": run_id,
        "scratchllm_version": scratchllm.__version__,
        "manifest_sha256": {Path(m).name: verify_manifest(Path(m)).sha256 for m in manifests},
        **summarize(index, neighbours, bucket_cap),
    }
    if run_id.startswith(TRACKED_PREFIX):  # raw web text must not enter git, see D-011
        for g in report["groups"].values():
            for ex in g["examples"]:
                ex["evaluation"].pop("start", None)
                ex["training"].pop("start", None)
    target = out_dir / run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "leakage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "leakage.md").write_text(render_markdown(report) + "\n", encoding="utf-8")
    return target


def fingerprint(index: Index) -> str:
    """A short hash of all signatures: equal inputs must give it on every run and machine."""
    if index.signatures is None or not len(index.signatures):
        return ""
    return hashlib.sha256(index.signatures.tobytes()).hexdigest()[:16]
