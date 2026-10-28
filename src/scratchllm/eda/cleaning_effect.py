"""What the cleaning rules do to a sample, measured before any rule is accepted.

Nothing is written: the rules run over the train split and the numbers come back. This is the
first part of acceptance criterion A1 (retention); the rest comes as more rules are added.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from scratchllm.data.cleaning import CleaningStats, clean_docs, rules_for
from scratchllm.data.sampling import verify_manifest
from scratchllm.data.sources import get_source
from scratchllm.eda.runner import iter_train_docs


def measure(manifest_path: Path) -> dict[str, Any]:
    manifest = verify_manifest(manifest_path)
    source = get_source(manifest.source["name"])
    rules = rules_for(source)
    lengths_before: list[int] = []
    stats = CleaningStats()

    def watch() -> Any:
        for doc in iter_train_docs(manifest_path):
            lengths_before.append(len(doc["text"]))
            yield doc

    lengths_after = [len(doc["text"]) for doc in clean_docs(watch(), rules, stats)]
    return {
        "source": source.name,
        "rules": [r.name for r in rules],
        "docs_in": stats.docs_in,
        "docs_out": stats.docs_out,
        "retention": round(stats.docs_out / stats.docs_in, 4) if stats.docs_in else 0.0,
        "chars_in": stats.chars_in,
        "chars_out": stats.chars_out,
        "median_chars_before": int(statistics.median(lengths_before)) if lengths_before else 0,
        "median_chars_after": int(statistics.median(lengths_after)) if lengths_after else 0,
        "share_under_300_before": _share_under(lengths_before, 300),
        "share_under_300_after": _share_under(lengths_after, 300),
        "dropped": dict(stats.dropped),
        "changed": dict(stats.changed),
        "chars_removed": dict(stats.chars_removed),
    }


def _share_under(lengths: list[int], limit: int) -> float:
    return round(sum(n < limit for n in lengths) / len(lengths), 4) if lengths else 0.0
